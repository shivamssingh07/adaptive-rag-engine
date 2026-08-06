"""Hybrid (BM25 + vector) retrieval via weighted score fusion.

Vector search alone misses exact keyword/entity matches (IDs, acronyms,
proper nouns embedded imprecisely); BM25 alone misses semantic paraphrase.
Combining both, weighted by `settings.hybrid_bm25_weight` /
`settings.hybrid_vector_weight` (which always sum to 1.0), is the default,
general-purpose retrieval strategy used across the system.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from langchain_core.documents import Document

from backend.config.settings import Settings, get_settings
from backend.core.exceptions import IndexNotFoundError
from backend.rag.retrievers.base import ScoredDocument, document_key
from backend.rag.retrievers.bm25_retriever import BM25RetrieverWrapper
from backend.rag.retrievers.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines BM25 lexical search and FAISS vector search via a weighted
    score ensemble."""

    def __init__(
        self,
        vector_retriever: VectorRetriever | None = None,
        bm25_retriever: BM25RetrieverWrapper | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Args:
        vector_retriever: Semantic retriever. Defaults to a fresh
            `VectorRetriever`.
        bm25_retriever: Lexical retriever. Defaults to a fresh
            `BM25RetrieverWrapper`.
        settings: Application settings. Defaults to the process-wide
            settings singleton.
        """
        self._vector = vector_retriever or VectorRetriever()
        self._bm25 = bm25_retriever or BM25RetrieverWrapper()
        self._settings = settings or get_settings()

    def retrieve(self, query: str, k: int | None = None) -> list[ScoredDocument]:
        """Retrieve and fuse results from both BM25 and vector search.

        Each source is over-fetched (3x `k`) before fusion so that
        documents ranked highly by only one source still have a fair
        chance of surviving into the final top-`k`. If one index is empty
        (e.g. only text has been ingested and BM25 legitimately returns
        nothing, or vice-versa), retrieval gracefully continues using only
        the available source rather than failing.

        Args:
            query: The search query text.
            k: Number of results to return. Defaults to
                `settings.top_k_retrieval`.

        Returns:
            Scored documents sorted by descending fused relevance.

        Raises:
            IndexNotFoundError: If *both* underlying indexes are empty.
        """
        effective_k = k or self._settings.top_k_retrieval
        fetch_k = max(effective_k * 3, effective_k)

        vector_results: list[ScoredDocument] = []
        bm25_results: list[ScoredDocument] = []

        try:
            vector_results = self._vector.retrieve(query, k=fetch_k)
        except IndexNotFoundError:
            logger.debug("Vector index is empty; hybrid retrieval will rely on BM25 only.")

        try:
            bm25_results = self._bm25.retrieve(query, k=fetch_k)
        except IndexNotFoundError:
            logger.debug("BM25 index is empty; hybrid retrieval will rely on vector search only.")

        if not vector_results and not bm25_results:
            raise IndexNotFoundError(
                "No documents have been indexed yet. Upload documents before querying."
            )

        combined_scores: dict[str, float] = defaultdict(float)
        document_lookup: dict[str, Document] = {}

        vector_weight = self._settings.hybrid_vector_weight
        bm25_weight = self._settings.hybrid_bm25_weight

        for scored in vector_results:
            key = document_key(scored.document)
            combined_scores[key] += vector_weight * scored.score
            document_lookup[key] = scored.document

        for scored in bm25_results:
            key = document_key(scored.document)
            combined_scores[key] += bm25_weight * scored.score
            document_lookup[key] = scored.document

        ranked = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
        ranked = ranked[:effective_k]
        return [ScoredDocument(document=document_lookup[key], score=score) for key, score in ranked]
