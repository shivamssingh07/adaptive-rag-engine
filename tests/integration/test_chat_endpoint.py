"""Integration tests for `POST /chat` and session management endpoints."""

from __future__ import annotations

import io


def _upload_sample_document(api_client) -> None:
    files = [
        (
            "files",
            (
                "policy.txt",
                io.BytesIO(b"Our refund policy allows returns within 30 days of purchase."),
                "text/plain",
            ),
        )
    ]
    api_client.post("/api/v1/upload", files=files)


class TestChatEndpointNonStreaming:
    def test_chat_returns_answer_with_citations_and_metrics(
        self, api_client, configure_llm_responses
    ) -> None:
        _upload_sample_document(api_client)
        configure_llm_responses(
            [
                "vectorstore",  # router
                "Our refund policy allows returns within 30 days.",  # compression
                "yes",  # grade_documents
                "Refunds are accepted within 30 days of purchase.",  # generate
                "yes",  # grade_generation
            ]
        )

        response = api_client.post(
            "/api/v1/chat",
            json={
                "message": "What is the company official refund policy for purchases?",
                "stream": False,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"]
        assert "session_id" in body
        assert body["metrics"]["route"] == "vectorstore"
        assert isinstance(body["sources"], list)

    def test_chat_with_no_documents_routes_direct_answer(
        self, api_client, configure_llm_responses
    ) -> None:
        configure_llm_responses(["Hello! How can I help you today?"])

        response = api_client.post("/api/v1/chat", json={"message": "Hi there!", "stream": False})

        assert response.status_code == 200
        body = response.json()
        assert body["metrics"]["route"] == "direct_answer"
        assert body["sources"] == []

    def test_chat_persists_session_across_turns(self, api_client, configure_llm_responses) -> None:
        _upload_sample_document(api_client)
        configure_llm_responses(
            [
                "vectorstore",
                "Our refund policy allows returns within 30 days.",
                "yes",
                "First answer.",
                "yes",
            ]
        )
        first = api_client.post(
            "/api/v1/chat",
            json={
                "message": "What is the company official refund policy for purchases?",
                "stream": False,
            },
        )
        session_id = first.json()["session_id"]

        configure_llm_responses(
            [
                "vectorstore",
                "Our refund policy allows returns within 30 days.",
                "yes",
                "Second answer.",
                "yes",
            ]
        )
        second = api_client.post(
            "/api/v1/chat",
            json={
                "message": "And what about returns from international customers?",
                "session_id": session_id,
                "stream": False,
            },
        )

        assert second.json()["session_id"] == session_id

    def test_empty_message_returns_validation_error(self, api_client) -> None:
        response = api_client.post("/api/v1/chat", json={"message": "", "stream": False})
        assert response.status_code == 422
        assert response.json()["error_code"] == "validation_error"


class TestChatEndpointStreaming:
    def test_streaming_produces_token_and_done_events(
        self, api_client, configure_llm_responses
    ) -> None:
        _upload_sample_document(api_client)
        configure_llm_responses(
            [
                "vectorstore",
                "Our refund policy allows returns within 30 days.",
                "yes",
                "Streaming works correctly for this answer.",
                "yes",
            ]
        )

        with api_client.stream(
            "POST",
            "/api/v1/chat",
            json={
                "message": "What is the company official refund policy for purchases?",
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())

        assert "event: token" in body
        assert "event: done" in body

        # Reconstruct the streamed answer from the SSE `token` events' JSON
        # payloads, rather than asserting on raw substring adjacency (the
        # answer is split across multiple separately-JSON-encoded chunks).
        import json

        reconstructed = ""
        for line in body.splitlines():
            if line.startswith("data:") and '"content"' in line:
                payload = json.loads(line.split(":", 1)[1].strip())
                reconstructed += payload["content"]
        assert reconstructed == "Streaming works correctly for this answer."


class TestSessionManagement:
    def test_export_session_transcript(self, api_client, configure_llm_responses) -> None:
        _upload_sample_document(api_client)
        configure_llm_responses(
            [
                "vectorstore",
                "Our refund policy allows returns within 30 days.",
                "yes",
                "Here is the answer.",
                "yes",
            ]
        )
        chat_response = api_client.post(
            "/api/v1/chat",
            json={
                "message": "What is the company official refund policy for purchases?",
                "stream": False,
            },
        )
        session_id = chat_response.json()["session_id"]

        export_response = api_client.get(f"/api/v1/chat/{session_id}/export")

        assert export_response.status_code == 200
        assert "refund policy" in export_response.json()["transcript"].lower()

    def test_clear_session(self, api_client, configure_llm_responses) -> None:
        _upload_sample_document(api_client)
        configure_llm_responses(
            [
                "vectorstore",
                "Our refund policy allows returns within 30 days.",
                "yes",
                "Answer.",
                "yes",
            ]
        )
        chat_response = api_client.post(
            "/api/v1/chat",
            json={
                "message": "What is the company official refund policy for purchases?",
                "stream": False,
            },
        )
        session_id = chat_response.json()["session_id"]

        clear_response = api_client.delete(f"/api/v1/chat/{session_id}")

        assert clear_response.status_code == 200

    def test_clear_nonexistent_session_returns_404(self, api_client) -> None:
        response = api_client.delete("/api/v1/chat/sess_does_not_exist")
        assert response.status_code == 404
        assert response.json()["error_code"] == "session_not_found"
