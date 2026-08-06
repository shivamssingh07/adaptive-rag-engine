"""Metrics panel component: renders per-turn observability badges
(latency, retrieval strategy, route, retries, token usage)."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_metrics_badges(metrics: dict[str, Any] | None, debug_mode: bool) -> None:
    """Render compact metrics badges beneath an assistant message.

    Args:
        metrics: The message's metrics dict (`ChatMetrics`), or `None` if
            unavailable.
        debug_mode: If `True`, also render the full raw metrics payload
            for debugging.
    """
    if not metrics:
        return

    caption_parts = [
        f"⏱️ {metrics['latency_ms']:.0f}ms",
        f"🧭 {metrics['route']}",
        f"🔍 {metrics['retrieval_strategy']}",
    ]
    token_usage = metrics.get("token_usage") or {}
    total_tokens = token_usage.get("total_tokens", 0)
    if total_tokens:
        caption_parts.append(f"🔢 {total_tokens} tokens")
    if metrics.get("used_web_search"):
        caption_parts.append("🌐 web search used")
    if metrics.get("document_relevance_retries"):
        caption_parts.append(f"🔁 {metrics['document_relevance_retries']} retrieval retr(y/ies)")
    if metrics.get("groundedness_retries"):
        caption_parts.append(f"🛡️ {metrics['groundedness_retries']} groundedness retr(y/ies)")

    st.caption(" · ".join(caption_parts))

    if debug_mode:
        with st.expander("🐛 Raw metrics (debug mode)"):
            st.json(metrics)
