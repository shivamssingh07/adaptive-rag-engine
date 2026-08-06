"""Config route: non-sensitive current configuration, for the Streamlit
Settings page. Never exposes API keys or other secrets."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_settings
from backend.api.schemas.config import ConfigResponse
from backend.config.settings import Settings

router = APIRouter(tags=["config"])


@router.get(
    "/config",
    response_model=ConfigResponse,
    summary="Current non-sensitive application configuration.",
)
async def get_config(settings: Settings = Depends(get_settings)) -> ConfigResponse:
    """Return the current configuration for display in the UI.

    Args:
        settings: Injected application settings.

    Returns:
        Non-sensitive configuration values only — API keys are never
        included in this response.
    """
    return ConfigResponse(
        app_name=settings.app_name,
        app_version=settings.app_version,
        environment=settings.environment,
        llm_model=settings.groq_model,
        embedding_model=settings.embedding_model_name,
        reranker_model=settings.reranker_model_name,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k_retrieval=settings.top_k_retrieval,
        top_k_rerank=settings.top_k_rerank,
        hybrid_bm25_weight=settings.hybrid_bm25_weight,
        hybrid_vector_weight=settings.hybrid_vector_weight,
        tavily_web_search_enabled=settings.tavily_enabled,
        max_document_grade_retries=settings.max_document_grade_retries,
        max_groundedness_retries=settings.max_groundedness_retries,
        allowed_extensions=sorted(settings.allowed_extensions),
        max_upload_size_mb=settings.max_upload_size_mb,
    )
