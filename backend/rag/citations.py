"""Source citation construction.

Turns raw retrieval results into a de-duplicated, client-ready citation
list — the shape the Streamlit UI and `/chat` API response (built in later
phases) both consume to show "sources" alongside a generated answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.rag.retrievers.base import ScoredDocument, document_key

_EXCERPT_MAX_CHARS = 240


@dataclass(slots=True)
class SourceCitation:
    """A single source citation attached to a generated answer."""

    source: str
    document_id: str | None
    chunk_id: str | None
    page: int | None
    row: int | None
    file_type: str | None
    score: float
    excerpt: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict, suitable for JSON API responses and
        for storing alongside a message in `SessionStore`.

        Returns:
            A JSON-serializable dict representation of this citation.
        """
        return {
            "source": self.source,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "page": self.page,
            "row": self.row,
            "file_type": self.file_type,
            "score": round(self.score, 4),
            "excerpt": self.excerpt,
        }


def _truncate_excerpt(text: str, max_chars: int = _EXCERPT_MAX_CHARS) -> str:
    """Collapse whitespace and truncate to a citation-friendly excerpt length."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    truncated = collapsed[:max_chars].rsplit(" ", 1)[0]
    return f"{truncated}..."


def build_citations(scored_documents: list[ScoredDocument]) -> list[SourceCitation]:
    """Build a de-duplicated, ranked list of source citations.

    Args:
        scored_documents: Scored retrieval results, typically the final
            reranked/compressed set used for generation, in descending
            relevance order.

    Returns:
        One `SourceCitation` per unique chunk (by `chunk_id`), preserving
        input order (i.e. already ranked by relevance).
    """
    citations: list[SourceCitation] = []
    seen: set[str] = set()

    for scored in scored_documents:
        key = document_key(scored.document)
        if key in seen:
            continue
        seen.add(key)

        metadata = scored.document.metadata
        citations.append(
            SourceCitation(
                source=str(metadata.get("source", "unknown")),
                document_id=metadata.get("document_id"),
                chunk_id=metadata.get("chunk_id"),
                page=metadata.get("page"),
                row=metadata.get("row"),
                file_type=metadata.get("file_type"),
                score=scored.score,
                excerpt=_truncate_excerpt(scored.document.page_content),
            )
        )

    return citations
