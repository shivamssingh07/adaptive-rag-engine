"""Abstract interface for embedding providers."""

from __future__ import annotations

from typing import Protocol

from langchain_core.embeddings import Embeddings


class EmbeddingProvider(Protocol):
    """Structural interface every embedding provider implementation must
    satisfy."""

    def get_embeddings(self) -> Embeddings:
        """Return a LangChain-compatible embeddings object.

        Returns:
            An object implementing `langchain_core.embeddings.Embeddings`
            (i.e. `embed_query` / `embed_documents`).
        """
        ...
