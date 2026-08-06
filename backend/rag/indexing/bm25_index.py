"""Persistent BM25 lexical index.

`rank-bm25`'s `BM25Okapi` has no native incremental-update or persistence
API, so this wrapper rebuilds the index from the full in-memory corpus on
every write and persists the whole thing (tokenized corpus + document
objects) to a single pickle file. This is a deliberate, documented
trade-off appropriate for a portfolio-scale corpus (thousands, not
millions, of chunks); a production system at larger scale would swap this
for a proper inverted-index search engine (e.g. Elasticsearch/OpenSearch)
without changing the `BM25RetrieverWrapper` interface that consumes it.
"""

from __future__ import annotations

import logging
import pickle
import re
import threading
from pathlib import Path

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from backend.config.settings import Settings, get_settings
from backend.core.exceptions import IndexNotFoundError, RetrievalError

logger = logging.getLogger(__name__)

_INDEX_FILE_NAME = "bm25_index.pkl"
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only tokenization used for both indexing
    and querying, so tokens match consistently on both sides."""
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    """Thread-safe, disk-persisted BM25 lexical index."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the index and attempt to reload it from disk.

        Args:
            settings: Application settings. Defaults to the process-wide
                settings singleton.
        """
        self._settings = settings or get_settings()
        self._bm25: BM25Okapi | None = None
        self._documents: list[Document] = []
        self._lock = threading.RLock()
        self._load_if_exists()

    def _index_file(self) -> Path:
        return self._settings.bm25_index_dir / _INDEX_FILE_NAME

    def _load_if_exists(self) -> None:
        path = self._index_file()
        if not path.exists():
            logger.info(
                "No existing BM25 index found at %s; it will be created on first upload.", path
            )
            return
        try:
            logger.info("Loading existing BM25 index from %s ...", path)
            with path.open("rb") as fh:
                payload = pickle.load(fh)  # noqa: S301 - trusted, self-written file
            self._bm25 = payload["bm25"]
            self._documents = payload["documents"]
            logger.info("BM25 index loaded successfully (%d documents).", len(self._documents))
        except Exception as exc:  # noqa: BLE001 - log and start fresh rather than crash
            logger.error("Failed to load existing BM25 index (%s); starting fresh.", exc)
            self._bm25 = None
            self._documents = []

    def _persist(self) -> None:
        path = self._index_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({"bm25": self._bm25, "documents": self._documents}, fh)
        logger.debug("BM25 index persisted to %s", path)

    @property
    def is_initialized(self) -> bool:
        """Whether the index has at least been created."""
        with self._lock:
            return self._bm25 is not None

    @property
    def document_count(self) -> int:
        """Total number of documents currently stored in the index."""
        with self._lock:
            return len(self._documents)

    def add_documents(self, documents: list[Document]) -> int:
        """Append documents to the corpus and rebuild the BM25 index.

        Args:
            documents: Chunked documents to add.

        Returns:
            The number of documents added.

        Raises:
            RetrievalError: If (re)building the index fails.
        """
        if not documents:
            return 0
        with self._lock:
            try:
                self._documents.extend(documents)
                tokenized_corpus = [_tokenize(doc.page_content) for doc in self._documents]
                self._bm25 = BM25Okapi(tokenized_corpus)
                self._persist()
            except Exception as exc:  # noqa: BLE001 - wrapped as a domain error
                raise RetrievalError(f"Failed to add documents to BM25 index: {exc}") from exc
        logger.info("Added %d document(s) to the BM25 index.", len(documents))
        return len(documents)

    def search(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        """Retrieve the `k` highest-BM25-scoring documents for `query`.

        Args:
            query: The search query text.
            k: Number of results to return. Defaults to
                `settings.top_k_retrieval`.

        Returns:
            A list of `(document, bm25_score)` tuples with score > 0,
            sorted descending. Documents with a score of exactly 0
            (no token overlap) are excluded.

        Raises:
            IndexNotFoundError: If no documents have been indexed yet.
            RetrievalError: If scoring fails.
        """
        with self._lock:
            if self._bm25 is None:
                raise IndexNotFoundError(
                    "No documents have been indexed yet. Upload documents before querying."
                )
            effective_k = k or self._settings.top_k_retrieval
            try:
                tokenized_query = _tokenize(query)
                scores = self._bm25.get_scores(tokenized_query)
            except Exception as exc:  # noqa: BLE001 - wrapped as a domain error
                raise RetrievalError(f"BM25 search failed: {exc}") from exc

            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
                :effective_k
            ]
            return [(self._documents[i], float(scores[i])) for i in ranked_indices if scores[i] > 0]

    def delete_by_document_id(self, document_id: str) -> int:
        """Remove every chunk belonging to a source document, then rebuild
        the BM25 index over the remaining corpus.

        Args:
            document_id: The `document_id` metadata value shared by every
                chunk of the document to remove.

        Returns:
            The number of chunks removed.

        Raises:
            RetrievalError: If rebuilding the index fails.
        """
        with self._lock:
            before_count = len(self._documents)
            remaining = [
                doc for doc in self._documents if doc.metadata.get("document_id") != document_id
            ]
            removed = before_count - len(remaining)
            if removed == 0:
                return 0
            try:
                self._documents = remaining
                if remaining:
                    self._bm25 = BM25Okapi([_tokenize(doc.page_content) for doc in remaining])
                else:
                    self._bm25 = None
                self._persist()
            except Exception as exc:  # noqa: BLE001 - wrapped as a domain error
                raise RetrievalError(
                    f"Failed to delete document '{document_id}' from BM25 index: {exc}"
                ) from exc
        logger.info(
            "Deleted %d chunk(s) belonging to document '%s' from the BM25 index.",
            removed,
            document_id,
        )
        return removed

    def clear(self) -> None:
        """Delete the in-memory index and any persisted file on disk."""
        with self._lock:
            self._bm25 = None
            self._documents = []
            self._index_file().unlink(missing_ok=True)
        logger.info("BM25 index cleared.")


_index_singleton: BM25Index | None = None
_index_lock = threading.Lock()


def get_bm25_index() -> BM25Index:
    """Return the process-wide `BM25Index` singleton.

    Returns:
        The shared `BM25Index` instance.
    """
    global _index_singleton
    if _index_singleton is None:
        with _index_lock:
            if _index_singleton is None:
                _index_singleton = BM25Index()
    return _index_singleton
