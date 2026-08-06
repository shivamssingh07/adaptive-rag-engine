"""Health check route.

Reports the configuration status of the application's key external
dependencies (Groq LLM, optional Tavily web search) without performing
expensive operations like actually loading the embedding or reranker
models — those are lazy-loaded on first use (see Phase 3) and are
intentionally *not* forced to load just to answer a health check, so this
endpoint stays fast and cheap enough to be polled frequently by an
orchestrator (Docker, Kubernetes, Render, Railway).
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_app_start_time, get_settings
from backend.api.schemas.common import ComponentStatus, HealthResponse, HealthStatus
from backend.config.settings import Settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Report application health and dependency configuration status.",
)
async def health_check(
    settings: Settings = Depends(get_settings),
    start_time: float = Depends(get_app_start_time),
) -> HealthResponse:
    """Return the current health status of the application.

    The overall status is:
        * ``HEALTHY``  — the required Groq LLM credential is configured.
        * ``DEGRADED`` — a required component is missing (currently only
          possible if the process somehow started without a valid
          `GROQ_API_KEY`, which `get_settings` normally prevents at
          startup).

    Args:
        settings: Injected application settings.
        start_time: Injected process start timestamp.

    Returns:
        A populated :class:`HealthResponse`.
    """
    groq_configured = bool(settings.groq_api_key.get_secret_value().strip())

    components = [
        ComponentStatus(
            name="groq_llm",
            configured=groq_configured,
            detail=f"model={settings.groq_model}",
        ),
        ComponentStatus(
            name="tavily_web_search",
            configured=settings.tavily_enabled,
            detail="optional fallback; disabled when TAVILY_API_KEY is unset",
        ),
        ComponentStatus(
            name="faiss_index_dir",
            configured=settings.faiss_index_dir.exists(),
            detail=str(settings.faiss_index_dir),
        ),
        ComponentStatus(
            name="bm25_index_dir",
            configured=settings.bm25_index_dir.exists(),
            detail=str(settings.bm25_index_dir),
        ),
    ]

    overall_status = HealthStatus.HEALTHY if groq_configured else HealthStatus.DEGRADED
    if not groq_configured:
        logger.warning("Health check reporting DEGRADED: Groq LLM is not configured.")

    return HealthResponse(
        status=overall_status,
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        uptime_seconds=round(time.time() - start_time, 3),
        components=components,
    )
