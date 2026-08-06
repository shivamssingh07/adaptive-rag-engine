"""Schemas for `GET /metrics`."""

from __future__ import annotations

from pydantic import BaseModel


class MetricsResponse(BaseModel):
    """Response body for `GET /metrics`. Aggregate, point-in-time system
    statistics — not a time-series metrics endpoint (there's no metrics
    store); suitable for a dashboard "at a glance" panel."""

    total_documents: int
    total_chunks_faiss: int
    total_chunks_bm25: int
    active_sessions: int
    embedding_model: str
    reranker_model: str
    llm_model: str
    tavily_web_search_enabled: bool
