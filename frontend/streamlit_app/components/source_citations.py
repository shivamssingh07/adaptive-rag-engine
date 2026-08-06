"""Source citations component: renders the `sources` list attached to an
assistant message as an expandable, ranked list with similarity scores."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_source_citations(sources: list[dict[str, Any]]) -> None:
    """Render source citations for one assistant message.

    Args:
        sources: The message's source citation dicts, as returned by
            `ChatResponse.sources` / the SSE `done` event's `sources` field.
    """
    if not sources:
        return

    with st.expander(f"📚 Sources ({len(sources)})"):
        for index, source in enumerate(sources, start=1):
            location = ""
            if source.get("page") is not None:
                location = f", page {source['page']}"
            elif source.get("row") is not None:
                location = f", row {source['row']}"

            confidence_pct = round(source.get("score", 0.0) * 100)
            st.markdown(
                f"**{index}. {source.get('source', 'unknown')}**{location} "
                f"— confidence {confidence_pct}%"
            )
            st.caption(f"“{source.get('excerpt', '')}”")
            if index < len(sources):
                st.markdown("---")
