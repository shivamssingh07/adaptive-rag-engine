"""Schemas for `POST /chat`."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for `POST /chat`."""

    message: str = Field(..., min_length=1, max_length=4000, description="The user's question.")
    session_id: str | None = Field(
        default=None,
        description="Existing session ID to continue a conversation. Omit to start a new one.",
    )
    stream: bool = Field(
        default=True,
        description="If true (default), respond via Server-Sent Events streaming. If false, "
        "return a single JSON response.",
    )


class SourceCitationSchema(BaseModel):
    """One source citation attached to a generated answer."""

    source: str
    document_id: str | None = None
    chunk_id: str | None = None
    page: int | None = None
    row: int | None = None
    file_type: str | None = None
    score: float
    excerpt: str


class ChatMetrics(BaseModel):
    """Observability metadata attached to every chat response."""

    latency_ms: float = Field(..., description="Total end-to-end processing time.")
    route: str = Field(
        ..., description="Adaptive routing decision: vectorstore/web_search/direct_answer."
    )
    retrieval_strategy: str = Field(
        ..., description="Adaptive retrieval strategy used: hybrid/mmr/multi_query/self_query/none."
    )
    used_web_search: bool
    document_relevance_retries: int
    groundedness_retries: int
    context_chunks_used: int
    token_usage: dict[str, int] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """Response body for `POST /chat` when `stream=false`."""

    session_id: str
    answer: str
    sources: list[SourceCitationSchema]
    metrics: ChatMetrics


class SessionExportResponse(BaseModel):
    """Response body for exporting a session's conversation transcript."""

    session_id: str
    transcript: str
