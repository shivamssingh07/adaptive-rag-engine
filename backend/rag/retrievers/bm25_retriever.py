"""Pure lexical (BM25) retrieval over the BM25 index."""

from __future__ import annotations

from backend.config.settings import Settings, get_settings
from backend.rag.indexing.bm25_index import BM25Index, get_bm25_index
from backend.rag.retrievers.base import ScoredDocument


class BM25RetrieverWrapper:
    """Retrieves chunks purely by lexical (keyword) overlap."""

    def __init__(
        self,
        bm25_index: BM25Index | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Args:
        bm25_index: Lexical index to search. Defaults to the process-wide
            singleton.
        settings: Application settings. Defaults to the process-wide
            settings singleton.
        """
        self._index = bm25_index or get_bm25_index()
        self._settings = settings or get_settings()

    def retrieve(self, query: str, k: int | None = None) -> list[ScoredDocument]:
        """Retrieve the `k` highest-BM25-scoring chunks for `query`.

        Raw BM25 scores are unbounded, so they are min-max normalized
        against the maximum score within this result set to produce a
        `[0, 1]`-ish score comparable with other retrievers (see
        `HybridRetriever`).

        Args:
            query: The search query text.
            k: Number of results to return. Defaults to
                `settings.top_k_retrieval`.

        Returns:
            Scored documents sorted by descending BM25 relevance.

        Raises:
            IndexNotFoundError: If no documents have been indexed yet.
            RetrievalError: If the underlying search fails.
        """
        effective_k = k or self._settings.top_k_retrieval
        results = self._index.search(query, k=effective_k)
        if not results:
            return []
        max_score = max(score for _, score in results) or 1.0
        return [ScoredDocument(document=doc, score=score / max_score) for doc, score in results]
