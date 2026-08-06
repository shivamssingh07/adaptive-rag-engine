"""Metrics route: aggregate, point-in-time system statistics."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.api.dependencies import (
    get_bm25_index,
    get_document_registry,
    get_faiss_store,
    get_session_store,
    get_settings,
)
from backend.api.schemas.metrics import MetricsResponse
from backend.config.settings import Settings
from backend.rag.indexing.bm25_index import BM25Index
from backend.rag.indexing.document_registry import DocumentRegistry
from backend.rag.indexing.faiss_store import FAISSVectorStore
from backend.rag.memory.session_store import SessionStore

router = APIRouter(tags=["metrics"])


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Aggregate system statistics: index sizes, active sessions, active models.",
)
async def get_metrics(
    faiss_store: FAISSVectorStore = Depends(get_faiss_store),
    bm25_index: BM25Index = Depends(get_bm25_index),
    registry: DocumentRegistry = Depends(get_document_registry),
    session_store: SessionStore = Depends(get_session_store),
    settings: Settings = Depends(get_settings),
) -> MetricsResponse:
    """Return current aggregate statistics for the Streamlit dashboard.

    Args:
        faiss_store: Injected vector store.
        bm25_index: Injected lexical index.
        registry: Injected document registry.
        session_store: Injected session store.
        settings: Injected application settings.

    Returns:
        Current index sizes, active session count, and active model names.
    """
    return MetricsResponse(
        total_documents=registry.document_count,
        total_chunks_faiss=faiss_store.document_count,
        total_chunks_bm25=bm25_index.document_count,
        active_sessions=session_store.session_count,
        embedding_model=settings.embedding_model_name,
        reranker_model=settings.reranker_model_name,
        llm_model=settings.groq_model,
        tavily_web_search_enabled=settings.tavily_enabled,
    )
