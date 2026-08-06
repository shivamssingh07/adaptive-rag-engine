"""Document registry.

Tracks document-level metadata (as opposed to the chunk-level metadata
already carried on each `Document.metadata` in FAISS/BM25) so the API can
answer "what documents have been uploaded?" and "delete document X"
without scanning the entire vector store, and so duplicate uploads (by
content hash) can be detected before spending time re-parsing and
re-embedding a file that's already indexed.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DocumentRecord:
    """Metadata about one uploaded source document."""

    document_id: str
    filename: str
    file_type: str
    file_hash: str
    chunk_count: int
    size_bytes: int
    uploaded_at: float


def compute_file_hash(file_path: Path) -> str:
    """Compute the SHA-256 hash of a file's contents, streamed in chunks
    so large uploads don't need to be read fully into memory.

    Args:
        file_path: Path to the file to hash.

    Returns:
        The hex-encoded SHA-256 digest.
    """
    hasher = hashlib.sha256()
    with file_path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


class DocumentRegistry:
    """Thread-safe, disk-persisted registry of uploaded document metadata."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Args:
        settings: Application settings. Defaults to the process-wide
            settings singleton.
        """
        self._settings = settings or get_settings()
        self._db_path: Path = self._settings.document_registry_db_path
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, check_same_thread=False)

    def _init_schema(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    uploaded_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash)"
            )
            connection.commit()
        logger.info("Document registry schema ready at %s", self._db_path)

    @staticmethod
    def _row_to_record(row: tuple[Any, ...]) -> DocumentRecord:
        return DocumentRecord(
            document_id=row[0],
            filename=row[1],
            file_type=row[2],
            file_hash=row[3],
            chunk_count=row[4],
            size_bytes=row[5],
            uploaded_at=row[6],
        )

    _SELECT_COLUMNS = (
        "document_id, filename, file_type, file_hash, chunk_count, size_bytes, uploaded_at"
    )

    def register(
        self,
        document_id: str,
        filename: str,
        file_type: str,
        file_hash: str,
        chunk_count: int,
        size_bytes: int,
    ) -> None:
        """Record a newly-ingested document's metadata.

        Args:
            document_id: The document's generated ID (shared by all of its
                chunks' `document_id` metadata).
            filename: Original uploaded filename.
            file_type: One of the loader-assigned file types (pdf, docx,
                text, markdown, csv).
            file_hash: SHA-256 hash of the file's contents (see
                `compute_file_hash`), used for duplicate detection.
            chunk_count: Number of chunks this document was split into.
            size_bytes: Original file size in bytes.
        """
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO documents "
                f"({self._SELECT_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (document_id, filename, file_type, file_hash, chunk_count, size_bytes, time.time()),
            )
            connection.commit()
        logger.info("Registered document '%s' (%s, %d chunks).", filename, document_id, chunk_count)

    def get_by_hash(self, file_hash: str) -> DocumentRecord | None:
        """Look up a document by content hash, for duplicate detection.

        Args:
            file_hash: SHA-256 hash to look up.

        Returns:
            The matching `DocumentRecord`, or `None` if no document with
            this content has been registered.
        """
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"SELECT {self._SELECT_COLUMNS} FROM documents WHERE file_hash = ? LIMIT 1",
                (file_hash,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def get(self, document_id: str) -> DocumentRecord | None:
        """Look up a document by ID.

        Args:
            document_id: The document ID to look up.

        Returns:
            The matching `DocumentRecord`, or `None` if it doesn't exist.
        """
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"SELECT {self._SELECT_COLUMNS} FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list_all(self) -> list[DocumentRecord]:
        """List every registered document, most recently uploaded first.

        Returns:
            All `DocumentRecord`s currently registered.
        """
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"SELECT {self._SELECT_COLUMNS} FROM documents ORDER BY uploaded_at DESC"
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def delete(self, document_id: str) -> bool:
        """Remove a document's registry entry.

        Note this only removes the *registry* row — callers are
        responsible for also removing the document's chunks from the
        FAISS and BM25 indexes (see `Indexer.delete_document`, which
        does both in the correct order).

        Args:
            document_id: The document to remove.

        Returns:
            `True` if a row was deleted, `False` if it didn't exist.
        """
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM documents WHERE document_id = ?", (document_id,)
            )
            connection.commit()
            deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Removed document '%s' from the registry.", document_id)
        return deleted

    def clear(self) -> None:
        """Remove every registered document (used by the `/reset` endpoint)."""
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM documents")
            connection.commit()
        logger.info("Document registry cleared.")

    @property
    def document_count(self) -> int:
        """Total number of registered documents."""
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM documents").fetchone()
        return int(row[0]) if row else 0


_registry_singleton: DocumentRegistry | None = None
_registry_lock = threading.Lock()


def get_document_registry() -> DocumentRegistry:
    """Return the process-wide `DocumentRegistry` singleton.

    Returns:
        The shared `DocumentRegistry` instance.
    """
    global _registry_singleton
    if _registry_singleton is None:
        with _registry_lock:
            if _registry_singleton is None:
                _registry_singleton = DocumentRegistry()
    return _registry_singleton
