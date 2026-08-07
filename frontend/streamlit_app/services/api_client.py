"""Typed HTTP client for the Adaptive RAG Engine backend.

The Streamlit frontend is a separate process from the FastAPI backend and
talks to it exclusively over HTTP — it never imports from `backend.*`
directly. This keeps the two deployable independently (matching the
architecture decision recorded in the Phase 1 design doc) and means the
frontend has no dependency on torch/FAISS/langgraph at all.

Base URL is configurable via the `API_BASE_URL` environment variable so
the same frontend image works against `http://backend:8000` in Docker
Compose or `http://localhost:8000` for local development.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx

DEFAULT_API_BASE_URL = "https://adaptive-rag-engine.onrender.com/api/v1"


class APIError(Exception):
    """Raised when the backend returns an error response.

    Carries the same `error_code`/`message` shape the backend's
    `ErrorResponse` envelope uses, so the UI can display a meaningful
    message instead of a raw HTTP status code.
    """

    def __init__(self, status_code: int, error_code: str, message: str) -> None:
        """Args:
        status_code: HTTP status code returned by the backend.
        error_code: The backend's stable error identifier.
        message: Human-readable error message.
        """
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        super().__init__(f"[{status_code}] {error_code}: {message}")


def _resolve_base_url() -> str:
    """Read the backend base URL from the environment, with a sane default
    for local development."""
    return os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


class APIClient:
    """Synchronous, typed HTTP client for every backend endpoint the
    Streamlit UI needs."""

    def __init__(self, base_url: str | None = None, timeout: float = 120.0) -> None:
        """Args:
        base_url: Backend API base URL. Defaults to `API_BASE_URL` env var
            or `http://localhost:8000/api/v1`.
        timeout: Request timeout in seconds. Generous default since
            document ingestion and multi-step graph runs can take a while.
        """
        self._base_url = base_url or _resolve_base_url()
        self._timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    @staticmethod
    def _raise_for_error(response: httpx.Response) -> None:
        """Translate a non-2xx response into an `APIError`.

        Args:
            response: The HTTP response to check.

        Raises:
            APIError: If `response.status_code >= 400`.
        """
        if response.status_code < 400:
            return
        response.read()  # ensure body is available even for streamed responses
        try:
            body: dict[str, Any] = response.json()
        except (json.JSONDecodeError, ValueError):
            body = {}
        raise APIError(
            status_code=response.status_code,
            error_code=body.get("error_code", "unknown_error"),
            message=body.get("message", response.text or "Unknown error"),
        )

    # ------------------------------------------------------------------
    # Health / config / metrics
    # ------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        """`GET /health` — application and dependency configuration status."""
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(self._url("/health"))
            self._raise_for_error(response)
            return response.json()

    def get_config(self) -> dict[str, Any]:
        """`GET /config` — current non-sensitive configuration."""
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(self._url("/config"))
            self._raise_for_error(response)
            return response.json()

    def get_metrics(self) -> dict[str, Any]:
        """`GET /metrics` — aggregate index/session statistics."""
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(self._url("/metrics"))
            self._raise_for_error(response)
            return response.json()

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------
    def list_documents(self) -> dict[str, Any]:
        """`GET /documents` — every currently-indexed document."""
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(self._url("/documents"))
            self._raise_for_error(response)
            return response.json()

    def upload_files(self, files: list[tuple[str, bytes, str]]) -> dict[str, Any]:
        """`POST /upload` — ingest one or more documents.

        Args:
            files: A list of `(filename, content_bytes, mime_type)` tuples.

        Returns:
            The upload batch summary.
        """
        multipart = [("files", (name, content, mime)) for name, content, mime in files]
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(self._url("/upload"), files=multipart)
            self._raise_for_error(response)
            return response.json()

    def delete_document(self, document_id: str) -> dict[str, Any]:
        """`DELETE /documents/{document_id}`."""
        with httpx.Client(timeout=self._timeout) as client:
            response = client.delete(self._url(f"/documents/{document_id}"))
            self._raise_for_error(response)
            return response.json()

    def reset_knowledge_base(self) -> dict[str, Any]:
        """`POST /reset` — clear the entire knowledge base."""
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(self._url("/reset"))
            self._raise_for_error(response)
            return response.json()

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------
    def chat(self, message: str, session_id: str | None = None) -> dict[str, Any]:
        """`POST /chat` (non-streaming) — one turn of the conversation.

        Args:
            message: The user's question.
            session_id: Existing session to continue, or `None` for a new one.

        Returns:
            The full `ChatResponse` payload.
        """
        payload = {"message": message, "session_id": session_id, "stream": False}
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(self._url("/chat"), json=payload)
            self._raise_for_error(response)
            return response.json()

    def chat_stream(self, message: str, session_id: str | None = None) -> Iterator[dict[str, Any]]:
        """`POST /chat` (streaming) — one turn of the conversation via SSE.

        Args:
            message: The user's question.
            session_id: Existing session to continue, or `None` for a new one.

        Yields:
            One dict per SSE frame: `{"event": "token", "content": "..."}`
            for each answer chunk, followed by exactly one
            `{"event": "done", "session_id": ..., "sources": [...], "metrics": {...}}`.
        """
        payload = {"message": message, "session_id": session_id, "stream": True}
        with (
            httpx.Client(timeout=self._timeout) as client,
            client.stream("POST", self._url("/chat"), json=payload) as response,
        ):
            self._raise_for_error(response)
            current_event = "message"
            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    raw_data = line.split(":", 1)[1].strip()
                    try:
                        parsed = json.loads(raw_data)
                    except json.JSONDecodeError:
                        continue
                    yield {"event": current_event, **parsed}

    def clear_session(self, session_id: str) -> dict[str, Any]:
        """`DELETE /chat/{session_id}` — clear a session's history."""
        with httpx.Client(timeout=self._timeout) as client:
            response = client.delete(self._url(f"/chat/{session_id}"))
            self._raise_for_error(response)
            return response.json()

    def export_session(self, session_id: str) -> dict[str, Any]:
        """`GET /chat/{session_id}/export` — full conversation transcript."""
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(self._url(f"/chat/{session_id}/export"))
            self._raise_for_error(response)
            return response.json()


def get_api_client() -> APIClient:
    """Return a new `APIClient` configured from the environment.

    A fresh client is cheap to construct (it holds no persistent
    connection; each method opens a short-lived `httpx.Client`), so this
    is called on every rerun rather than cached in `st.session_state`.
    """
    return APIClient()
