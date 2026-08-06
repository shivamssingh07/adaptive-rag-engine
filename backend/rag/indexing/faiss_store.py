"""Persistent FAISS vector store.

Wraps `langchain_community.vectorstores.FAISS` with:
    * Automatic reload from disk on startup if a previously-persisted
      index exists at `settings.faiss_index_dir`.
    * Automatic persistence to disk after every write.
    * Thread-safety via an `RLock` (FAISS's in-memory index is not
      inherently thread-safe for concurrent writes).
    * Domain exceptions instead of leaking raw FAISS/LangChain errors.

Deserializing a LangChain FAISS index from disk uses Python's `pickle`
internally for the document store component. `allow_dangerous_deserialization=True`
is safe here because the index is always one this application wrote itself
to a local, non-user-supplied path — it is never loaded from an
untrusted/external source.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from backend.config.settings import Settings, get_settings
from backend.core.exceptions import IndexNotFoundError, RetrievalError
from backend.rag.embeddings.huggingface_provider import get_huggingface_provider

logger = logging.getLogger(__name__)

_FAISS_INDEX_FILE = "index.faiss"


class FAISSVectorStore:
    """Thread-safe, disk-persisted wrapper around a LangChain FAISS index."""

    def __init__(
        self,
        settings: Settings | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        """Initialize the store and attempt to reload a persisted index.

        Args:
            settings: Application settings. Defaults to the process-wide
                settings singleton.
            embeddings: Embeddings implementation used both to build the
                index and to embed queries at search time. Defaults to the
                process-wide HuggingFace embedding provider.
        """
        self._settings = settings or get_settings()
        self._embeddings = embeddings or get_huggingface_provider().get_embeddings()
        self._store: FAISS | None = None
        self._lock = threading.RLock()
        self._load_if_exists()

    def _index_dir(self) -> Path:
        return self._settings.faiss_index_dir

    def _load_if_exists(self) -> None:
        """Reload a previously-persisted index from disk, if present."""
        index_file = self._index_dir() / _FAISS_INDEX_FILE
        if not index_file.exists():
            logger.info(
                "No existing FAISS index found at %s; it will be created on first upload.",
                self._index_dir(),
            )
            return
        try:
            logger.info("Loading existing FAISS index from %s ...", self._index_dir())
            self._store = FAISS.load_local(
                folder_path=str(self._index_dir()),
                embeddings=self._embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("FAISS index loaded successfully (%d vectors).", self._store.index.ntotal)
        except Exception as exc:  # noqa: BLE001 - log and start fresh rather than crash
            logger.error(
                "Failed to load existing FAISS index (%s); starting with an empty index.", exc
            )
            self._store = None

    def _persist(self) -> None:
        assert self._store is not None
        self._index_dir().mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self._index_dir()))
        logger.debug("FAISS index persisted to %s", self._index_dir())

    @property
    def is_initialized(self) -> bool:
        """Whether the index has at least been created (documents may
        still be zero if it was just created empty)."""
        with self._lock:
            return self._store is not None

    @property
    def document_count(self) -> int:
        """Total number of vectors currently stored in the index."""
        with self._lock:
            return int(self._store.index.ntotal) if self._store is not None else 0

    def add_documents(self, documents: list[Document]) -> int:
        """Embed and add documents to the index, then persist to disk.

        Args:
            documents: Chunked documents (see `backend.rag.loaders.splitter`)
                to embed and index.

        Returns:
            The number of documents added.

        Raises:
            RetrievalError: If embedding or indexing fails.
        """
        if not documents:
            return 0
        with self._lock:
            try:
                if self._store is None:
                    self._store = FAISS.from_documents(documents, self._embeddings)
                else:
                    self._store.add_documents(documents)
                self._persist()
            except Exception as exc:  # noqa: BLE001 - wrapped as a domain error
                raise RetrievalError(f"Failed to add documents to FAISS index: {exc}") from exc
        logger.info("Added %d document(s) to the FAISS index.", len(documents))
        return len(documents)

    def similarity_search_with_score(
        self,
        query: str,
        k: int | None = None,
        filter: dict[str, str] | None = None,  # noqa: A002 - matches LangChain's parameter name
    ) -> list[tuple[Document, float]]:
        """Retrieve the `k` nearest chunks to `query` by embedding distance.

        Args:
            query: The search query text.
            k: Number of results to return. Defaults to
                `settings.top_k_retrieval`.
            filter: Optional metadata equality filter (e.g.
                `{"source": "report.pdf"}`), used by self-query retrieval.

        Returns:
            A list of `(document, l2_distance)` tuples — lower distance
            means more similar. Callers that want a bounded similarity
            score should transform this (see `VectorRetriever`).

        Raises:
            IndexNotFoundError: If no documents have been indexed yet.
            RetrievalError: If the underlying search fails.
        """
        with self._lock:
            if self._store is None:
                raise IndexNotFoundError(
                    "No documents have been indexed yet. Upload documents before querying."
                )
            effective_k = k or self._settings.top_k_retrieval
            try:
                return self._store.similarity_search_with_score(query, k=effective_k, filter=filter)
            except Exception as exc:  # noqa: BLE001 - wrapped as a domain error
                raise RetrievalError(f"FAISS similarity search failed: {exc}") from exc

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int | None = None,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> list[Document]:
        """Retrieve `k` chunks balancing relevance and diversity (MMR).

        Args:
            query: The search query text.
            k: Number of results to return. Defaults to
                `settings.top_k_retrieval`.
            fetch_k: Number of candidates fetched by initial similarity
                search before the MMR re-selection is applied.
            lambda_mult: Trade-off between relevance (1.0) and diversity
                (0.0).

        Returns:
            A list of `Document` objects.

        Raises:
            IndexNotFoundError: If no documents have been indexed yet.
            RetrievalError: If the underlying search fails.
        """
        with self._lock:
            if self._store is None:
                raise IndexNotFoundError(
                    "No documents have been indexed yet. Upload documents before querying."
                )
            effective_k = k or self._settings.top_k_retrieval
            try:
                return self._store.max_marginal_relevance_search(
                    query, k=effective_k, fetch_k=fetch_k, lambda_mult=lambda_mult
                )
            except Exception as exc:  # noqa: BLE001 - wrapped as a domain error
                raise RetrievalError(f"FAISS MMR search failed: {exc}") from exc

    def delete_by_document_id(self, document_id: str) -> int:
        """Remove every chunk belonging to a source document from the index.

        LangChain's `InMemoryDocstore` (the FAISS wrapper's internal
        document store) does not expose a public "find IDs by metadata"
        API, so this reaches into its `_dict` attribute to enumerate
        stored documents. This is a stable, widely-relied-upon pattern in
        the LangChain ecosystem for this exact gap, not an unsupported
        internal detail specific to this project.

        Args:
            document_id: The `document_id` metadata value shared by every
                chunk of the document to remove.

        Returns:
            The number of chunks removed.

        Raises:
            RetrievalError: If the underlying delete operation fails.
        """
        with self._lock:
            if self._store is None:
                return 0
            # LangChain's `InMemoryDocstore` doesn't expose a public API to
            # enumerate stored documents by metadata; `_dict` is a stable,
            # widely-relied-upon access pattern for this exact gap.
            docstore_items = self._store.docstore._dict.items()  # type: ignore[attr-defined]  # noqa: SLF001
            ids_to_delete = [
                doc_id
                for doc_id, doc in docstore_items
                if doc.metadata.get("document_id") == document_id
            ]
            if not ids_to_delete:
                return 0
            try:
                self._store.delete(ids_to_delete)
                self._persist()
            except Exception as exc:  # noqa: BLE001 - wrapped as a domain error
                raise RetrievalError(
                    f"Failed to delete document '{document_id}' from FAISS index: {exc}"
                ) from exc
        logger.info(
            "Deleted %d chunk(s) belonging to document '%s' from the FAISS index.",
            len(ids_to_delete),
            document_id,
        )
        return len(ids_to_delete)

    def clear(self) -> None:
        """Delete the in-memory index and any persisted files on disk."""
        with self._lock:
            self._store = None
            for artifact in self._index_dir().glob("index.*"):
                artifact.unlink(missing_ok=True)
        logger.info("FAISS index cleared.")


_store_singleton: FAISSVectorStore | None = None
_store_lock = threading.Lock()


def get_faiss_store() -> FAISSVectorStore:
    """Return the process-wide `FAISSVectorStore` singleton.

    Returns:
        The shared `FAISSVectorStore` instance.
    """
    global _store_singleton
    if _store_singleton is None:
        with _store_lock:
            if _store_singleton is None:
                _store_singleton = FAISSVectorStore()
    return _store_singleton
