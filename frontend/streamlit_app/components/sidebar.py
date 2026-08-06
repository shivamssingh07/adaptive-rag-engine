"""Sidebar component: navigation, live index stats, and destructive actions
(clear chat, reset knowledge base)."""

from __future__ import annotations

import streamlit as st

from frontend.streamlit_app.services.api_client import APIClient, APIError
from frontend.streamlit_app.state.session_state import clear_chat_display, reset_session

_PAGES = ["Chat", "Documents", "Settings", "About"]


def render_sidebar(client: APIClient) -> None:
    """Render the sidebar: branding, navigation, live stats, and actions.

    Args:
        client: Backend API client.
    """
    with st.sidebar:
        st.markdown("## 🧠 Adaptive RAG Engine")
        st.caption("Self-correcting, hybrid-retrieval RAG — 100% free stack.")

        st.session_state["page"] = st.radio(
            "Navigate",
            _PAGES,
            index=_PAGES.index(st.session_state["page"]),
            label_visibility="collapsed",
        )

        st.divider()
        _render_live_stats(client)

        st.divider()
        st.markdown("#### Session")
        if st.session_state["session_id"]:
            st.caption(f"Session: `{st.session_state['session_id']}`")
        else:
            st.caption("No active session yet — send a message to start one.")

        if st.button("🧹 Clear chat", use_container_width=True):
            clear_chat_display()
            st.rerun()

        if st.button("🔄 New session", use_container_width=True):
            reset_session()
            st.rerun()

        st.divider()
        with st.expander("⚠️ Danger zone"):
            st.caption("Permanently deletes every indexed document. Cannot be undone.")
            confirm = st.checkbox("I understand this cannot be undone", key="confirm_reset")
            if st.button("🗑️ Reset knowledge base", disabled=not confirm, use_container_width=True):
                try:
                    result = client.reset_knowledge_base()
                    st.success(result.get("message", "Knowledge base reset."))
                    st.session_state["confirm_reset"] = False
                    st.rerun()
                except APIError as exc:
                    st.error(f"Reset failed: {exc.message}")

        st.divider()
        st.session_state["debug_mode"] = st.toggle(
            "🐛 Debug mode",
            value=st.session_state["debug_mode"],
            help="Show raw retrieval scores, routing decisions, and per-turn metrics.",
        )
        st.caption(
            "Prefer a dark theme? Use the ☰ menu (top-right) → Settings → Theme, "
            "Streamlit's native dark mode."
        )


def _render_live_stats(client: APIClient) -> None:
    """Render a compact knowledge-base stats panel, tolerant of the
    backend being unreachable (e.g. not started yet)."""
    st.markdown("#### Knowledge base")
    try:
        metrics = client.get_metrics()
        col1, col2 = st.columns(2)
        col1.metric("Documents", metrics["total_documents"])
        col2.metric("Chunks", metrics["total_chunks_faiss"])
        st.caption(f"LLM: `{metrics['llm_model']}`")
        st.caption(f"Embeddings: `{metrics['embedding_model'].split('/')[-1]}`")
        if metrics["tavily_web_search_enabled"]:
            st.caption("🌐 Web search fallback: enabled")
        else:
            st.caption("🌐 Web search fallback: disabled")
    except APIError as exc:
        st.warning(f"Backend error: {exc.message}")
    except Exception:  # noqa: BLE001 - backend may simply not be reachable yet
        st.warning("⚠️ Cannot reach the backend API. Is it running?")
