"""Schemas for `GET /config`."""

from __future__ import annotations

from pydantic import BaseModel


class ConfigResponse(BaseModel):
    """Response body for `GET /config`. Non-sensitive current
    configuration, safe to display in the Streamlit "Settings" page.
    Never includes API keys or other secrets."""

    app_name: str
    app_version: str
    environment: str
    llm_model: str
    embedding_model: str
    reranker_model: str
    chunk_size: int
    chunk_overlap: int
    top_k_retrieval: int
    top_k_rerank: int
    hybrid_bm25_weight: float
    hybrid_vector_weight: float
    tavily_web_search_enabled: bool
    max_document_grade_retries: int
    max_groundedness_retries: int
    allowed_extensions: list[str]
    max_upload_size_mb: int
