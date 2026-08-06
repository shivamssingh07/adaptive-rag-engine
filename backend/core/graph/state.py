"""State schema for the adaptive RAG LangGraph state machine.

A Pydantic model is used (rather than a bare `TypedDict`) so a node
returning the wrong shape for a field fails fast with a validation error
during development, instead of silently corrupting state deep into a run.
Every node function receives a `GraphState` instance and returns a `dict`
of the fields it updates — LangGraph merges that dict into the running
state between node executions.
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field

from backend.rag.retrievers.base import ScoredDocument


class GraphState(BaseModel):
    """The full state threaded through every node of the adaptive RAG graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- Input ------------------------------------------------------------
    session_id: str
    original_question: str
    history_text: str = ""

    # --- Routing ------------------------------------------------------------
    route: str = ""

    # --- Retrieval ------------------------------------------------------------
    rewritten_query: str = ""
    retrieval_strategy: str = ""
    retrieved_documents: list[ScoredDocument] = Field(default_factory=list)
    compressed_documents: list[Document] = Field(default_factory=list)
    used_web_search: bool = False

    # --- Self-correction ------------------------------------------------------------
    relevance_grade: str = ""
    groundedness_grade: str = ""
    retry_count_documents: int = 0
    retry_count_groundedness: int = 0

    # --- Output ------------------------------------------------------------
    generation: str = ""
    token_usage: dict[str, int] = Field(default_factory=dict)

    # --- Observability (populated by the API layer around graph.invoke) -----
    timings_ms: dict[str, float] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def effective_query(self) -> str:
        """The query text retrieval should actually use: the rewritten
        query if one was produced, otherwise the original question."""
        return self.rewritten_query or self.original_question
