"""Optional Tavily web-search fallback.

Used by the graph's `web_search` node when local retrieval is unavailable
or graded irrelevant after retries. Fully optional: every method returns
gracefully (empty results, not an exception) when `TAVILY_API_KEY` is
unset, so the rest of the system never needs to special-case "is web
search configured?" beyond checking `settings.tavily_enabled` once, up
front, in the graph's routing logic.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from langchain_core.documents import Document
from tavily import TavilyClient

from backend.config.settings import Settings, get_settings
from backend.core.exceptions import ExternalServiceError

logger = logging.getLogger(__name__)


class TavilySearchProvider:
    """Thread-safe, lazily-initialized wrapper around the Tavily search API."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Args:
        settings: Application settings. Defaults to the process-wide
            settings singleton.
        """
        self._settings = settings or get_settings()
        self._client: TavilyClient | None = None
        self._lock = threading.Lock()

    def _get_client(self) -> TavilyClient:
        """Return the Tavily client, constructing it on first call.

        Raises:
            ExternalServiceError: If Tavily is not configured, or the
                client cannot be constructed.
        """
        if not self._settings.tavily_enabled:
            raise ExternalServiceError(
                "Tavily web search was requested but TAVILY_API_KEY is not configured."
            )
        if self._client is None:
            with self._lock:
                if self._client is None:
                    logger.info("Initializing Tavily web-search client...")
                    try:
                        assert self._settings.tavily_api_key is not None
                        self._client = TavilyClient(
                            api_key=self._settings.tavily_api_key.get_secret_value()
                        )
                    except Exception as exc:  # noqa: BLE001 - wrapped as a domain error
                        raise ExternalServiceError(
                            f"Failed to initialize Tavily client: {exc}"
                        ) from exc
        return self._client

    def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Run a web search and return results as LangChain documents.

        Returns an empty list (rather than raising) when Tavily is not
        configured, or when the search itself fails — callers (the graph's
        `web_search` node) treat "no web results" as a valid, gracefully
        degraded outcome, not a fatal error.

        Args:
            query: The search query text.
            max_results: Maximum number of web results to return.

        Returns:
            A list of `Document` objects, one per web result with usable
            content, tagged with `file_type="web"` metadata.
        """
        if not self._settings.tavily_enabled:
            logger.debug("Tavily is not configured; returning no web results.")
            return []

        try:
            client = self._get_client()
            response: dict[str, Any] = client.search(query=query, max_results=max_results)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.warning("Tavily web search failed (%s); returning no web results.", exc)
            return []

        raw_results = response.get("results", []) if isinstance(response, dict) else []
        documents: list[Document] = []
        for item in raw_results:
            content = str(item.get("content") or item.get("raw_content") or "").strip()
            if not content:
                continue
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": item.get("url", "web"),
                        "title": item.get("title", ""),
                        "file_type": "web",
                    },
                )
            )

        logger.info("Tavily web search returned %d usable result(s) for query.", len(documents))
        return documents


_provider_singleton: TavilySearchProvider | None = None
_provider_lock = threading.Lock()


def get_tavily_search() -> TavilySearchProvider:
    """Return the process-wide `TavilySearchProvider` singleton.

    Returns:
        The shared `TavilySearchProvider` instance.
    """
    global _provider_singleton
    if _provider_singleton is None:
        with _provider_lock:
            if _provider_singleton is None:
                _provider_singleton = TavilySearchProvider()
    return _provider_singleton
