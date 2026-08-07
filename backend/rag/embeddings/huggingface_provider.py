"""Lightweight FastEmbed embedding provider.

Uses FastEmbed/ONNX instead of sentence-transformers/PyTorch so the
application can run within low-memory environments such as Render Free.

The default model remains:
    sentence-transformers/all-MiniLM-L6-v2

This model produces 384-dimensional embeddings.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from langchain_core.embeddings import Embeddings

from backend.config.settings import Settings, get_settings
from backend.core.exceptions import ExternalServiceError

if TYPE_CHECKING:
    from fastembed import TextEmbedding

logger = logging.getLogger(__name__)


class FastEmbedEmbeddings(Embeddings):
    """LangChain-compatible wrapper around FastEmbed TextEmbedding."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: TextEmbedding | None = None
        self._lock = threading.Lock()

    def _get_model(self) -> TextEmbedding:
        """Lazy-load the FastEmbed ONNX model."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from fastembed import TextEmbedding

                        logger.info(
                            "Loading lightweight FastEmbed model '%s'...",
                            self.model_name,
                        )

                        self._model = TextEmbedding(
                            model_name=self.model_name,
                        )

                        logger.info(
                            "FastEmbed model '%s' loaded successfully.",
                            self.model_name,
                        )

                    except Exception as exc:  # noqa: BLE001
                        raise ExternalServiceError(
                            f"Failed to load FastEmbed model '{self.model_name}': {exc}",
                            details={"model": self.model_name},
                        ) from exc

        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for documents."""
        if not texts:
            return []

        try:
            model = self._get_model()

            vectors = model.embed(
                texts,
                batch_size=8,
            )

            return [vector.tolist() for vector in vectors]

        except ExternalServiceError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ExternalServiceError(f"Failed to embed documents: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a query."""
        try:
            model = self._get_model()

            vector = next(model.query_embed([text]))

            return [float(value) for value in vector.tolist()]

        except ExternalServiceError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ExternalServiceError(f"Failed to embed query: {exc}") from exc


class HuggingFaceEmbeddingProvider:
    """Thread-safe, lazily initialized lightweight embedding provider."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings: Embeddings | None = None
        self._dimension: int | None = None
        self._lock = threading.Lock()

    def get_embeddings(self) -> Embeddings:
        """Return the cached FastEmbed embedding model."""
        if self._embeddings is None:
            with self._lock:
                if self._embeddings is None:
                    model_name = self._settings.embedding_model_name

                    logger.info(
                        "Initializing lightweight FastEmbed provider with model '%s'...",
                        model_name,
                    )

                    try:
                        self._embeddings = FastEmbedEmbeddings(model_name=model_name)
                    except Exception as exc:  # noqa: BLE001
                        raise ExternalServiceError(
                            f"Failed to initialize embedding provider: {exc}",
                            details={"model": model_name},
                        ) from exc

        return self._embeddings

    @property
    def embedding_dimension(self) -> int:
        """Return embedding vector dimensionality."""
        if self._dimension is None:
            probe_vector = self.get_embeddings().embed_query("dimension probe")
            self._dimension = len(probe_vector)

        return self._dimension


_provider_singleton: HuggingFaceEmbeddingProvider | None = None
_provider_lock = threading.Lock()


def get_huggingface_provider() -> HuggingFaceEmbeddingProvider:
    """Return the process-wide embedding provider singleton."""
    global _provider_singleton

    if _provider_singleton is None:
        with _provider_lock:
            if _provider_singleton is None:
                _provider_singleton = HuggingFaceEmbeddingProvider()

    return _provider_singleton
