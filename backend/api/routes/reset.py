"""Reset route: clear the entire knowledge base (all indexed documents)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_bm25_index, get_document_registry, get_faiss_store
from backend.api.schemas.documents import ResetResponse
from backend.rag.indexing.bm25_index import BM25Index
from backend.rag.indexing.document_registry import DocumentRegistry
from backend.rag.indexing.faiss_store import FAISSVectorStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["reset"])


@router.post(
    "/reset",
    response_model=ResetResponse,
    summary="Clear the entire knowledge base (all indexed documents, FAISS and BM25).",
)
async def reset_knowledge_base(
    faiss_store: FAISSVectorStore = Depends(get_faiss_store),
    bm25_index: BM25Index = Depends(get_bm25_index),
    registry: DocumentRegistry = Depends(get_document_registry),
) -> ResetResponse:
    """Clear the FAISS index, BM25 index, and document registry entirely.

    This does not affect conversation history — use
    `DELETE /chat/{session_id}` to clear a specific conversation. This is
    a deliberately destructive, irreversible operation with no
    confirmation step at the API layer; the Streamlit frontend is
    responsible for confirming with the user before calling this.

    Args:
        faiss_store: Injected vector store.
        bm25_index: Injected lexical index.
        registry: Injected document registry.

    Returns:
        A confirmation message with the number of documents that were removed.
    """
    documents_removed = registry.document_count
    faiss_store.clear()
    bm25_index.clear()
    registry.clear()
    logger.warning("Knowledge base reset: %d document(s) removed.", documents_removed)
    return ResetResponse(
        message="Knowledge base reset successfully. All indexed documents have been removed.",
        documents_removed=documents_removed,
    )
