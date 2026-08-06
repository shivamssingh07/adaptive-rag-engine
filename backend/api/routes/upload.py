"""Upload route.

`POST /upload` accepts one or more files, saves each under
`settings.upload_dir` (preserving the original filename so citations show
readable source names), and runs the full ingestion pipeline
(`backend.rag.indexing.indexer.Indexer`) on the batch. Failures are
isolated per-file — one corrupted file in a 10-file batch never prevents
the other 9 from being indexed.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.api.dependencies import get_indexer, get_settings
from backend.api.schemas.upload import FileIngestionResultSchema, UploadResponse
from backend.config.settings import Settings
from backend.rag.indexing.indexer import Indexer

logger = logging.getLogger(__name__)

router = APIRouter(tags=["upload"])


async def _save_upload(upload: UploadFile, upload_dir: Path) -> Path | None:
    """Persist one uploaded file to disk under a unique subdirectory.

    A unique subdirectory (rather than a unique filename prefix) is used
    so the original filename is preserved exactly — loaders use
    `file_path.name` as the `source` metadata shown in citations, and a
    mangled name like `a1b2c3_report.pdf` would look wrong to the user.

    Args:
        upload: The incoming `UploadFile`.
        upload_dir: Base directory uploads are stored under.

    Returns:
        The path the file was saved to, or `None` if `upload` had no
        filename (a malformed multipart part).
    """
    if not upload.filename:
        return None
    safe_dir = upload_dir / uuid.uuid4().hex
    safe_dir.mkdir(parents=True, exist_ok=True)
    destination = safe_dir / Path(upload.filename).name
    content = await upload.read()
    destination.write_bytes(content)
    return destination


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload one or more documents (PDF, DOCX, TXT, Markdown, CSV) for indexing.",
)
async def upload_documents(
    files: list[UploadFile] = File(..., description="One or more files to ingest."),
    indexer: Indexer = Depends(get_indexer),
    settings: Settings = Depends(get_settings),
) -> UploadResponse:
    """Save and ingest a batch of uploaded files.

    Args:
        files: The uploaded files.
        indexer: Injected ingestion pipeline.
        settings: Injected application settings.

    Returns:
        A summary of the batch, including per-file success/failure detail.
    """
    saved_paths: list[Path] = []
    for upload in files:
        saved_path = await _save_upload(upload, settings.upload_dir)
        if saved_path is not None:
            saved_paths.append(saved_path)

    summary = await run_in_threadpool(indexer.ingest_batch, saved_paths)

    logger.info(
        "Upload batch complete: %d/%d file(s) succeeded, %d chunk(s) added.",
        summary.successful_files,
        summary.total_files,
        summary.total_chunks_added,
    )

    return UploadResponse(
        total_files=summary.total_files,
        successful_files=summary.successful_files,
        failed_files=summary.failed_files,
        total_chunks_added=summary.total_chunks_added,
        results=[
            FileIngestionResultSchema(
                filename=r.filename,
                success=r.success,
                chunks_added=r.chunks_added,
                document_id=r.document_id,
                duplicate=r.duplicate,
                error=r.error,
            )
            for r in summary.results
        ],
    )
