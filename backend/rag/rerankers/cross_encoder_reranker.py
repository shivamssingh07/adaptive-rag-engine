"""Optional cross-encoder reranking.

The cross-encoder is used as a second-stage reranker over a small candidate
set returned by initial retrieval.

For low-memory deployments such as Render Free, reranking can be disabled by
setting:

    RERANKER_MODEL_NAME=

When disabled, the original retrieval order is preserved and no
cross-encoder model is loaded.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from backend.config.settings import Settings, get_settings
from backend.core.exceptions import ExternalServiceError, RetrievalError

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Thread-safe, lazily initialized optional cross-encoder reranker."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the reranker without loading the model."""
        self._settings = settings or get_settings()
        self._model: CrossEncoder | None = None
        self._lock = threading.Lock()

    def _get_model(self) -> CrossEncoder:
        """Return the cross-encoder model, loading it only when required."""
        model_name = self._settings.reranker_model_name

        if not model_name or not model_name.strip():
            raise ExternalServiceError(
                "Cross-encoder reranker is disabled because "
                "RERANKER_MODEL_NAME is empty."
            )

        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info(
                        "Loading cross-encoder reranker model '%s' "
                        "(downloaded automatically on first use, then cached)...",
                        model_name,
                    )

                    try:
                        from sentence_transformers import CrossEncoder

                        self._model = CrossEncoder(model_name)
                    except Exception as exc:  # noqa: BLE001
                        raise ExternalServiceError(
                            f"Failed to load reranker model '{model_name}': {exc}",
                            details={"model": model_name},
                        ) from exc

                    logger.info(
                        "Cross-encoder reranker model loaded successfully."
                    )

        return self._model

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """Re-score and re-order candidate documents against the query.

        When RERANKER_MODEL_NAME is empty, reranking is skipped and the
        original retrieval order is preserved.
        """
        if not documents:
            return []

        k = top_k if top_k is not None else self._settings.top_k_rerank

        if k <= 0:
            return []

        # --------------------------------------------------------------
        # Low-memory mode
        # --------------------------------------------------------------
        # On Render Free, set RERANKER_MODEL_NAME to an empty value.
        # This prevents sentence-transformers and the CrossEncoder model
        # from being loaded at all.
        model_name = self._settings.reranker_model_name

        if not model_name or not model_name.strip():
            logger.info(
                "Cross-encoder reranking disabled. "
                "Returning first-stage retrieval results."
            )

            return [
                (document, 0.0)
                for document in documents[:k]
            ]

        # --------------------------------------------------------------
        # Normal reranking mode
        # --------------------------------------------------------------
        model = self._get_model()

        pairs = [
            (query, document.page_content)
            for document in documents
        ]

        try:
            raw_scores = model.predict(pairs)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(
                f"Cross-encoder reranking failed: {exc}"
            ) from exc

        scored = list(
            zip(
                documents,
                (float(score) for score in raw_scores),
                strict=True,
            )
        )

        # Highest relevance score first.
        scored.sort(
            key=lambda pair: pair[1],
            reverse=True,
        )

        return scored[:k]


# ----------------------------------------------------------------------
# Process-wide singleton
# ----------------------------------------------------------------------

_reranker_singleton: CrossEncoderReranker | None = None
_reranker_lock = threading.Lock()


def get_cross_encoder_reranker() -> CrossEncoderReranker:
    """Return the process-wide CrossEncoderReranker singleton."""
    global _reranker_singleton

    if _reranker_singleton is None:
        with _reranker_lock:
            if _reranker_singleton is None:
                _reranker_singleton = CrossEncoderReranker()

    return _reranker_singleton