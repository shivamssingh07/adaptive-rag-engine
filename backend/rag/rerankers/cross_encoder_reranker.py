"""Cross-encoder reranking.

A bi-encoder (the embedding model used for initial retrieval) scores query
and document independently, which is fast but less precise. A
cross-encoder scores the (query, document) pair jointly, which is far more
accurate but too slow to run over an entire corpus — so it's used here as a
second-stage reranker over the small candidate set returned by initial
retrieval, using `BAAI/bge-reranker-base` via `sentence-transformers`.
"""

from __future__ import annotations

import logging
import threading

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from backend.config.settings import Settings, get_settings
from backend.core.exceptions import ExternalServiceError, RetrievalError

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Thread-safe, lazily-initialized cross-encoder reranker."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the reranker without loading the model.

        Args:
            settings: Application settings. Defaults to the process-wide
                settings singleton.
        """
        self._settings = settings or get_settings()
        self._model: CrossEncoder | None = None
        self._lock = threading.Lock()

    def _get_model(self) -> CrossEncoder:
        """Return the cross-encoder model, loading it on first call.

        Returns:
            The cached `CrossEncoder` instance.

        Raises:
            ExternalServiceError: If the model fails to download or load.
        """
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info(
                        "Loading cross-encoder reranker model '%s' "
                        "(downloaded automatically on first use, then cached)...",
                        self._settings.reranker_model_name,
                    )
                    try:
                        self._model = CrossEncoder(self._settings.reranker_model_name)
                    except Exception as exc:  # noqa: BLE001 - wrapped as a domain error
                        raise ExternalServiceError(
                            f"Failed to load reranker model "
                            f"'{self._settings.reranker_model_name}': {exc}",
                            details={"model": self._settings.reranker_model_name},
                        ) from exc
                    logger.info("Cross-encoder reranker model loaded successfully.")
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """Re-score and re-order candidate documents against the query.

        Args:
            query: The user's search query.
            documents: Candidate documents from a first-stage retriever
                (typically 10-30 chunks).
            top_k: Number of top-scoring documents to return. Defaults to
                `settings.top_k_rerank`.

        Returns:
            A list of `(document, relevance_score)` tuples, sorted by
            descending relevance, truncated to `top_k`.

        Raises:
            RetrievalError: If scoring fails.
        """
        if not documents:
            return []

        model = self._get_model()
        pairs = [(query, doc.page_content) for doc in documents]

        try:
            raw_scores = model.predict(pairs)
        except Exception as exc:  # noqa: BLE001 - wrapped as a domain error
            raise RetrievalError(f"Cross-encoder reranking failed: {exc}") from exc

        scored = list(zip(documents, (float(score) for score in raw_scores), strict=True))
        scored.sort(key=lambda pair: pair[1], reverse=True)

        k = top_k if top_k is not None else self._settings.top_k_rerank
        return scored[:k]


_reranker_singleton: CrossEncoderReranker | None = None
_reranker_lock = threading.Lock()


def get_cross_encoder_reranker() -> CrossEncoderReranker:
    """Return the process-wide `CrossEncoderReranker` singleton.

    Returns:
        The shared `CrossEncoderReranker` instance.
    """
    global _reranker_singleton
    if _reranker_singleton is None:
        with _reranker_lock:
            if _reranker_singleton is None:
                _reranker_singleton = CrossEncoderReranker()
    return _reranker_singleton
