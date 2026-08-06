"""Schemas for `POST /upload`."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FileIngestionResultSchema(BaseModel):
    """Outcome of ingesting a single file within an upload batch."""

    filename: str
    success: bool
    chunks_added: int
    document_id: str | None = None
    duplicate: bool = False
    error: str | None = None


class UploadResponse(BaseModel):
    """Response body for `POST /upload`."""

    total_files: int = Field(..., description="Number of files submitted in this batch.")
    successful_files: int
    failed_files: int
    total_chunks_added: int
    results: list[FileIngestionResultSchema]
