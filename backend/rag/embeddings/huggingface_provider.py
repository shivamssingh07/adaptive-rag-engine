"""HuggingFace embedding provider.

Wraps `langchain_huggingface.HuggingFaceEmbeddings` (backed by
`sentence-transformers/all-MiniLM-L6-v2` by default) behind a thread-safe,
lazily-initialized singleton. The model weights are downloaded
automatically from the HuggingFace Hub on first use and cached locally by
`sentence-transformers` in the standard HF cache directory — no manual
download step is required, and no API key or paid service is involved.
"""

from __future__ import annotations

import logging
import threading

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from backend.config.settings import Settings, get_settings
from backend.core.exceptions import ExternalServiceError

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddingProvider:
    """Thread-safe, lazily-initialized wrapper around a local HuggingFace
    sentence-embedding model."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the provider without loading the model.

        Args:
            settings: Application settings. Defaults to the process-wide
                settings singleton.
        """
        self._settings = settings or get_settings()
        self._embeddings: Embeddings | None = None
        self._dimension: int | None = None
        self._lock = threading.Lock()

    def get_embeddings(self) -> Embeddings:
        """Return the embedding model, loading it from HuggingFace on first
        call and caching it for the lifetime of the process.

        Returns:
            The cached `HuggingFaceEmbeddings` instance.

        Raises:
            ExternalServiceError: If the model fails to download or load
                (e.g. no network access, invalid model name).
        """
        if self._embeddings is None:
            with self._lock:
                if self._embeddings is None:
                    logger.info(
                        "Loading HuggingFace embedding model '%s' "
                        "(downloaded automatically on first use, then cached)...",
                        self._settings.embedding_model_name,
                    )
                    try:
                        self._embeddings = HuggingFaceEmbeddings(
                            model_name=self._settings.embedding_model_name,
                            encode_kwargs={"normalize_embeddings": True},
                        )
                    except Exception as exc:  # noqa: BLE001 - wrapped as a domain error
                        raise ExternalServiceError(
                            f"Failed to load embedding model "
                            f"'{self._settings.embedding_model_name}': {exc}",
                            details={"model": self._settings.embedding_model_name},
                        ) from exc
                    logger.info("Embedding model loaded successfully.")
        return self._embeddings

    @property
    def embedding_dimension(self) -> int:
        """The output vector dimensionality of the loaded embedding model.

        Determined lazily by embedding a short probe string on first
        access, then cached — avoids hardcoding a dimension that would
        silently go stale if `EMBEDDING_MODEL_NAME` is changed.

        Returns:
            The embedding vector length (e.g. 384 for MiniLM-L6-v2).
        """
        if self._dimension is None:
            probe_vector = self.get_embeddings().embed_query("dimension probe")
            self._dimension = len(probe_vector)
        return self._dimension


_provider_singleton: HuggingFaceEmbeddingProvider | None = None
_provider_lock = threading.Lock()


def get_huggingface_provider() -> HuggingFaceEmbeddingProvider:
    """Return the process-wide `HuggingFaceEmbeddingProvider` singleton.

    Returns:
        The shared `HuggingFaceEmbeddingProvider` instance.
    """
    global _provider_singleton
    if _provider_singleton is None:
        with _provider_lock:
            if _provider_singleton is None:
                _provider_singleton = HuggingFaceEmbeddingProvider()
    return _provider_singleton
