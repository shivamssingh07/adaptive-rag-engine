"""Chat panel component: renders the conversation transcript and handles
new message submission with streamed assistant responses."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import streamlit as st

from frontend.streamlit_app.components.metrics_panel import render_metrics_badges
from frontend.streamlit_app.components.source_citations import render_source_citations
from frontend.streamlit_app.services.api_client import APIClient, APIError
from frontend.streamlit_app.state.session_state import add_message


def render_chat_panel(client: APIClient) -> None:
    """Render the full chat panel: history, then the input box.

    Args:
        client: Backend API client.
    """
    _render_history()

    prompt = st.chat_input("Ask a question about your documents...")
    if prompt:
        _handle_new_message(client, prompt)


def _render_history() -> None:
    """Render every message already in `st.session_state["messages"]`."""
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_source_citations(message["sources"])
                render_metrics_badges(message["metrics"], st.session_state["debug_mode"])


def _handle_new_message(client: APIClient, prompt: str) -> None:
    """Submit a new user message, stream the assistant's response, and
    persist both to the local transcript.

    Args:
        client: Backend API client.
        prompt: The user's message text.
    """
    add_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    result: dict[str, Any] = {}

    with st.chat_message("assistant"):
        try:
            full_answer = st.write_stream(
                _token_generator(client, prompt, st.session_state["session_id"], result)
            )
        except APIError as exc:
            st.error(f"Chat failed: {exc.message}")
            return
        except Exception as exc:  # noqa: BLE001 - surface connectivity issues plainly
            st.error(f"Could not reach the backend: {exc}")
            return

        sources = result.get("sources", [])
        metrics = result.get("metrics")
        render_source_citations(sources)
        render_metrics_badges(metrics, st.session_state["debug_mode"])

    if "session_id" in result:
        st.session_state["session_id"] = result["session_id"]

    add_message("assistant", full_answer, sources=sources, metrics=metrics)


def _token_generator(
    client: APIClient, prompt: str, session_id: str | None, result_holder: dict[str, Any]
) -> Iterator[str]:
    """Adapt `APIClient.chat_stream`'s SSE event stream into a plain text
    generator for `st.write_stream`, capturing the final `done` event's
    metadata into `result_holder` as a side effect.

    Args:
        client: Backend API client.
        prompt: The user's message text.
        session_id: Existing session to continue, or `None` for a new one.
        result_holder: Mutable dict populated with the `done` event's
            payload once streaming completes — read by the caller after
            `st.write_stream` returns.

    Yields:
        Answer text chunks, in order.
    """
    for event in client.chat_stream(prompt, session_id):
        if event.get("event") == "token":
            yield event.get("content", "")
        elif event.get("event") == "done":
            result_holder.update(event)
