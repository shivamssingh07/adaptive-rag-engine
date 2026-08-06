"""Pure semantic (vector) retrieval over the FAISS index."""

from __future__ import annotations

from backend.config.settings import Settings, get_settings
from backend.rag.indexing.faiss_store import FAISSVectorStore, get_faiss_store
from backend.rag.retrievers.base import ScoredDocument


class VectorRetriever:
    """Retrieves chunks purely by embedding similarity."""

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

    def retrieve(self, query: str, k: int | None = None) -> list[ScoredDocument]:
        """Retrieve the `k` most semantically similar chunks to `query`.

        FAISS (configured with the default Euclidean distance strategy)
        returns L2 distance, where lower means more similar and the value
        is unbounded. It is converted here to a bounded, descending
        similarity score via `1 / (1 + distance)` so it can be combined
        fairly with other retrievers' scores (see `HybridRetriever`).

        Args:
            query: The search query text.
            k: Number of results to return. Defaults to
                `settings.top_k_retrieval`.

        Returns:
            Scored documents sorted by descending similarity.

        Raises:
            IndexNotFoundError: If no documents have been indexed yet.
            RetrievalError: If the underlying search fails.
        """
        effective_k = k or self._settings.top_k_retrieval
        results = self._store.similarity_search_with_score(query, k=effective_k)
        return [
            ScoredDocument(document=doc, score=1.0 / (1.0 + distance)) for doc, distance in results
        ]
