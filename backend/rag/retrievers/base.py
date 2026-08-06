"""Shared types used across every retriever implementation."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass(slots=True)
class ScoredDocument:
    """A retrieved document paired with a relevance score.

    Every retriever in this package normalizes its native scoring scheme
    (FAISS L2 distance, BM25 term-frequency score, cross-encoder logits,
    MMR rank) into this common shape so downstream code — hybrid fusion,
    citation building, the graph's relevance grader — can treat results
    from any retriever uniformly. Higher `score` always means more
    relevant.
    """

    document: Document
    score: float

    def __post_init__(self) -> None:
        # FAISS/BM25/numpy computations frequently produce numpy scalar
        # types (e.g. numpy.float32) rather than a native Python float.
        # Coercing here — once, centrally — prevents that from silently
        # leaking into `json.dumps` calls downstream (API responses,
        # session persistence), which raise on numpy scalars.
        self.score = float(self.score)


def document_key(document: Document) -> str:
    """Stable de-duplication key for a chunk, used when merging results
    from multiple retrieval sources (hybrid fusion, multi-query merging).

    Prefers the chunk's deterministic `chunk_id` metadata; falls back to a
    content prefix for documents that somehow lack it (e.g. in tests).
    """
    chunk_id = document.metadata.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    return document.page_content[:200]
