"""Schemas for `GET /documents` and `DELETE /documents/{document_id}`."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    """Metadata about one uploaded source document."""

    document_id: str
    filename: str
    file_type: str
    chunk_count: int
    size_bytes: int
    uploaded_at: datetime


class DocumentListResponse(BaseModel):
    """Response body for `GET /documents`."""

    total_documents: int
    total_chunks: int
    documents: list[DocumentInfo]


class DeleteDocumentResponse(BaseModel):
    """Response body for `DELETE /documents/{document_id}`."""

    document_id: str
    deleted: bool
    chunks_removed: int = Field(..., description="Chunks removed from both FAISS and BM25.")


class ResetResponse(BaseModel):
    """Response body for `POST /reset`."""

    message: str
    documents_removed: int
