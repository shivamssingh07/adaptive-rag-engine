"""Maximal Marginal Relevance (MMR) retrieval.

Standard similarity search can return several near-duplicate chunks (e.g.
five chunks all covering the same paragraph from slightly different
angles), which wastes context budget. MMR re-selects from a larger
candidate pool to balance relevance against diversity, which is
particularly useful for broad, summary-style questions.
"""

from __future__ import annotations

from backend.config.settings import Settings, get_settings
from backend.rag.indexing.faiss_store import FAISSVectorStore, get_faiss_store
from backend.rag.retrievers.base import ScoredDocument


class MMRRetriever:
    """Retrieves chunks using FAISS's Maximal Marginal Relevance search."""

    def __init__(
        self,
        faiss_store: FAISSVectorStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Args:
        faiss_store: Vector store to search. Defaults to the process-wide
            singleton.
        settings: Application settings. Defaults to the process-wide
            settings singleton.
        """
        self._store = faiss_store or get_faiss_store()
        self._settings = settings or get_settings()

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> list[ScoredDocument]:
        """Retrieve `k` chunks balancing relevance and diversity.

        MMR does not produce a native relevance score (it returns an
        already-diversified, relevance-ordered list), so a synthetic
        descending rank-based score is assigned here purely so the return
        type stays consistent with every other retriever in this package.

        Args:
            query: The search query text.
            k: Number of results to return. Defaults to
                `settings.top_k_retrieval`.
            fetch_k: Size of the initial similarity-search candidate pool
                MMR selects from.
            lambda_mult: Trade-off between relevance (1.0) and diversity
                (0.0).

        Returns:
            Scored documents in MMR-selected order (first = best).

        Raises:
            IndexNotFoundError: If no documents have been indexed yet.
            RetrievalError: If the underlying search fails.
        """
        effective_k = k or self._settings.top_k_retrieval
        documents = self._store.max_marginal_relevance_search(
            query, k=effective_k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )
        total = len(documents)
        return [
            ScoredDocument(document=doc, score=(total - index) / total)
            for index, doc in enumerate(documents)
        ]
