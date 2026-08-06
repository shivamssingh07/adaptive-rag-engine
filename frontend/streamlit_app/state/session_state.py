"""Streamlit `session_state` initialization and helpers.

Kept separate from rendering code so every page can call `init_state()`
idempotently at the top of a run without duplicating initialization logic
or accidentally clobbering existing state on rerun.
"""

from __future__ import annotations

from typing import Any, TypedDict

import streamlit as st


class ChatMessage(TypedDict):
    """One rendered message in the chat transcript."""

    role: str  # "user" | "assistant"
    content: str
    sources: list[dict[str, Any]]
    metrics: dict[str, Any] | None


def init_state() -> None:
    """Initialize every `st.session_state` key used across the app, if not
    already present. Safe to call on every script rerun."""
    defaults: dict[str, Any] = {
        "session_id": None,
        "messages": [],
        "page": "Chat",
        "debug_mode": False,
        "last_upload_summary": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_message(
    role: str,
    content: str,
    sources: list[dict[str, Any]] | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    """Append a message to the rendered chat transcript.

    Args:
        role: `"user"` or `"assistant"`.
        content: Message text.
        sources: Source citations (assistant messages only).
        metrics: Observability metrics (assistant messages only).
    """
    message: ChatMessage = {
        "role": role,
        "content": content,
        "sources": sources or [],
        "metrics": metrics,
    }
    st.session_state["messages"].append(message)


def clear_chat_display() -> None:
    """Clear the locally-rendered chat transcript.

    This only clears what's shown in the UI — it does not call the
    backend. Pair with `APIClient.clear_session` to also clear the
    server-side history, or leave the session as-is to keep server-side
    memory intact for context while giving the user a visually fresh view.
    """
    st.session_state["messages"] = []


def reset_session() -> None:
    """Fully reset to a brand-new conversation: clears the transcript and
    drops the session ID so the next message starts a new backend session."""
    st.session_state["messages"] = []
    st.session_state["session_id"] = None
