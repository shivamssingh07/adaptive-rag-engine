"""Document management routes: list and delete individually-indexed documents."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from backend.api.dependencies import get_document_registry, get_indexer
from backend.api.schemas.documents import DeleteDocumentResponse, DocumentInfo, DocumentListResponse
from backend.core.exceptions import DocumentNotFoundError
from backend.rag.indexing.document_registry import DocumentRegistry
from backend.rag.indexing.indexer import Indexer

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List every currently-indexed document.",
)
async def list_documents(
    registry: DocumentRegistry = Depends(get_document_registry),
) -> DocumentListResponse:
    """List all indexed documents with their metadata.

    Args:
        registry: Injected document registry.

    Returns:
        Every registered document, most recently uploaded first.
    """
    records = registry.list_all()
    return DocumentListResponse(
        total_documents=len(records),
        total_chunks=sum(record.chunk_count for record in records),
        documents=[
            DocumentInfo(
                document_id=record.document_id,
                filename=record.filename,
                file_type=record.file_type,
                chunk_count=record.chunk_count,
                size_bytes=record.size_bytes,
                uploaded_at=record.uploaded_at,  # type: ignore[arg-type]
            )
            for record in records
        ],
    )


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteDocumentResponse,
    summary="Delete a single document and all of its chunks from both indexes.",
)
async def delete_document(
    document_id: str,
    indexer: Indexer = Depends(get_indexer),
    registry: DocumentRegistry = Depends(get_document_registry),
) -> DeleteDocumentResponse:
    """Delete one document's chunks from FAISS and BM25, and its registry entry.

    Args:
        document_id: The document to delete.
        indexer: Injected ingestion/deletion pipeline.
        registry: Injected document registry.

    Returns:
        Confirmation of the deletion, including chunks removed.

    Raises:
        DocumentNotFoundError: If `document_id` does not exist.
    """
    if registry.get(document_id) is None:
        raise DocumentNotFoundError(
            f"Document '{document_id}' does not exist.", details={"document_id": document_id}
        )

    chunks_removed = await run_in_threadpool(indexer.delete_document, document_id)
    return DeleteDocumentResponse(
        document_id=document_id, deleted=True, chunks_removed=chunks_removed
    )
