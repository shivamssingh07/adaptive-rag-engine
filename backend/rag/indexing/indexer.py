"""Ingestion pipeline orchestration.

`Indexer` wires together, in order: file validation → format-specific
loading → chunking → writing to both the FAISS and BM25 indexes. Each file
in a batch is processed independently and failures are isolated per-file
(a corrupted PDF in a 10-file batch upload does not prevent the other 9
from being indexed) — see `ingest_batch`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from backend.config.settings import Settings, get_settings
from backend.core.exceptions import (
    DocumentProcessingError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from backend.rag.indexing.bm25_index import BM25Index, get_bm25_index
from backend.rag.indexing.document_registry import (
    DocumentRegistry,
    compute_file_hash,
    get_document_registry,
)
from backend.rag.indexing.faiss_store import FAISSVectorStore, get_faiss_store
from backend.rag.loaders.registry import get_loader_for
from backend.rag.loaders.splitter import DocumentSplitter

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FileIngestionResult:
    """Outcome of ingesting a single file."""

    filename: str
    success: bool
    chunks_added: int = 0
    error: str | None = None
    document_id: str | None = None
    duplicate: bool = False


@dataclass(slots=True)
class IngestionSummary:
    """Aggregate outcome of a (possibly multi-file) ingestion batch."""

    total_files: int
    successful_files: int
    failed_files: int
    total_chunks_added: int
    results: list[FileIngestionResult] = field(default_factory=list)


class Indexer:
    """Orchestrates the full ingestion pipeline: validate → load → split →
    index (FAISS + BM25)."""

    def __init__(
        self,
        settings: Settings | None = None,
        faiss_store: FAISSVectorStore | None = None,
        bm25_index: BM25Index | None = None,
        splitter: DocumentSplitter | None = None,
        document_registry: DocumentRegistry | None = None,
    ) -> None:
        """Wire up the indexer's dependencies.

        Args:
            settings: Application settings. Defaults to the process-wide
                settings singleton.
            faiss_store: Vector store to write chunks to. Defaults to the
                process-wide singleton.
            bm25_index: Lexical index to write chunks to. Defaults to the
                process-wide singleton.
            splitter: Chunking strategy. Defaults to a fresh
                `DocumentSplitter` built from `settings`.
            document_registry: Document-level metadata store, used for
                duplicate detection and to power document listing/deletion.
                Defaults to the process-wide singleton.
        """
        self._settings = settings or get_settings()
        self._faiss_store = faiss_store or get_faiss_store()
        self._bm25_index = bm25_index or get_bm25_index()
        self._splitter = splitter or DocumentSplitter(self._settings)
        self._document_registry = document_registry or get_document_registry()

    def _validate_file(self, file_path: Path) -> None:
        """Check extension and size before attempting to parse.

        Raises:
            UnsupportedFileTypeError: If the extension isn't allow-listed.
            FileTooLargeError: If the file exceeds the configured size cap.
        """
        extension = file_path.suffix.lower()
        if extension not in self._settings.allowed_extensions:
            raise UnsupportedFileTypeError(
                f"'{extension}' is not a supported file type.",
                details={
                    "extension": extension,
                    "allowed": sorted(self._settings.allowed_extensions),
                },
            )
        size_bytes = file_path.stat().st_size
        if size_bytes > self._settings.max_upload_size_bytes:
            raise FileTooLargeError(
                f"File '{file_path.name}' "
                f"({size_bytes / (1024 * 1024):.1f} MB) exceeds the "
                f"{self._settings.max_upload_size_mb} MB limit.",
                details={"file": file_path.name, "size_bytes": size_bytes},
            )

    def ingest_file(self, file_path: Path) -> FileIngestionResult:
        """Ingest a single file: validate, check for duplicates, load,
        split, and index it.

        Never raises — all failures (validation, parsing, indexing) are
        caught and returned as a failed `FileIngestionResult`, so this
        method is always safe to call in a loop over a batch.

        If a file with identical content (by SHA-256 hash) has already
        been ingested, indexing is skipped entirely and the existing
        document's ID is returned with `duplicate=True` — this both saves
        redundant parsing/embedding work and prevents the same content
        from cluttering retrieval results twice under two different
        `document_id`s.

        Args:
            file_path: Path to the file on disk (e.g. a temp upload path).

        Returns:
            The outcome of ingesting this file.
        """
        try:
            self._validate_file(file_path)

            file_hash = compute_file_hash(file_path)
            existing = self._document_registry.get_by_hash(file_hash)
            if existing is not None:
                logger.info(
                    "Skipping '%s': identical content already indexed as '%s' (%s).",
                    file_path.name,
                    existing.filename,
                    existing.document_id,
                )
                return FileIngestionResult(
                    filename=file_path.name,
                    success=True,
                    chunks_added=0,
                    document_id=existing.document_id,
                    duplicate=True,
                )

            loader = get_loader_for(file_path)
            raw_documents = loader.load(file_path)
            chunks = self._splitter.split(raw_documents)
            self._faiss_store.add_documents(chunks)
            self._bm25_index.add_documents(chunks)

            document_id = str(raw_documents[0].metadata.get("document_id"))
            file_type = str(raw_documents[0].metadata.get("file_type", "unknown"))
            self._document_registry.register(
                document_id=document_id,
                filename=file_path.name,
                file_type=file_type,
                file_hash=file_hash,
                chunk_count=len(chunks),
                size_bytes=file_path.stat().st_size,
            )

            logger.info("Ingested '%s': %d chunk(s) indexed.", file_path.name, len(chunks))
            return FileIngestionResult(
                filename=file_path.name,
                success=True,
                chunks_added=len(chunks),
                document_id=document_id,
            )
        except DocumentProcessingError as exc:
            logger.warning("Skipping '%s': %s", file_path.name, exc.message)
            return FileIngestionResult(filename=file_path.name, success=False, error=exc.message)
        except Exception as exc:  # noqa: BLE001 - isolate failures per file in a batch
            logger.error("Unexpected error ingesting '%s': %s", file_path.name, exc, exc_info=True)
            return FileIngestionResult(filename=file_path.name, success=False, error=str(exc))

    def delete_document(self, document_id: str) -> int:
        """Remove a document's chunks from both indexes and its registry entry.

        Args:
            document_id: The document to remove.

        Returns:
            The number of chunks removed from the FAISS index (the FAISS
            and BM25 counts are always equal since both are written
            together in `ingest_file`; FAISS's count is authoritative for
            the return value).
        """
        faiss_removed = self._faiss_store.delete_by_document_id(document_id)
        bm25_removed = self._bm25_index.delete_by_document_id(document_id)
        self._document_registry.delete(document_id)
        logger.info(
            "Deleted document '%s': %d chunk(s) removed from FAISS, %d from BM25.",
            document_id,
            faiss_removed,
            bm25_removed,
        )
        return faiss_removed

    def ingest_batch(self, file_paths: list[Path]) -> IngestionSummary:
        """Ingest multiple files, isolating failures per-file.

        Args:
            file_paths: Paths to the files to ingest.

        Returns:
            An `IngestionSummary` with per-file results and aggregate
            counts.
        """
        results = [self.ingest_file(file_path) for file_path in file_paths]
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        summary = IngestionSummary(
            total_files=len(results),
            successful_files=len(successful),
            failed_files=len(failed),
            total_chunks_added=sum(r.chunks_added for r in successful),
            results=results,
        )
        logger.info(
            "Batch ingestion complete: %d/%d file(s) succeeded, %d chunk(s) added.",
            summary.successful_files,
            summary.total_files,
            summary.total_chunks_added,
        )
        return summary
