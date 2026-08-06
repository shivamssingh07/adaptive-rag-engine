"""FastAPI application entry point.

Run directly for local development:

    python -m backend.api.main

Or via uvicorn (used by the Docker image and recommended for anything
beyond local development):

    uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.dependencies import get_settings
from backend.api.middleware.error_handler import register_exception_handlers
from backend.api.middleware.logging_middleware import LoggingMiddleware
from backend.api.middleware.request_id import RequestIDMiddleware
from backend.api.routes.chat import router as chat_router
from backend.api.routes.config import router as config_router
from backend.api.routes.documents import router as documents_router
from backend.api.routes.health import router as health_router
from backend.api.routes.metrics import router as metrics_router
from backend.api.routes.reset import router as reset_router
from backend.api.routes.upload import router as upload_router
from backend.config.constants import API_V1_PREFIX
from backend.config.logging_config import configure_logging

# Configure logging as early as possible — before the app object is even
# built — so that any errors during app construction itself are still
# captured by our structured logging setup rather than falling back to
# Python's default unconfigured behavior.
configure_logging(get_settings())
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown lifecycle hook.

    Startup:
        * Re-applies logging configuration (idempotent) in case settings
          were overridden (e.g. by tests) between import time and app
          startup.
        * Records the process start time on `app.state` for the health
          endpoint's uptime calculation.
        * Logs a startup banner summarizing key configuration so a
          deployer can immediately see what's active in the logs.

    Shutdown:
        * Logs a shutdown notice. Later phases will additionally flush any
          in-memory session state and close the FAISS/BM25 index handles
          cleanly here.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control back to FastAPI to serve requests until shutdown begins.
    """
    settings = get_settings()
    configure_logging(settings)
    app.state.start_time = time.time()
    app.state.settings = settings

    logger.info(
        "Starting %s v%s [environment=%s, debug=%s]",
        settings.app_name,
        settings.app_version,
        settings.environment,
        settings.debug,
    )
    logger.info("Groq model: %s", settings.groq_model)
    tavily_status = "enabled" if settings.tavily_enabled else "disabled"
    logger.info("Tavily web-search fallback: %s", tavily_status)
    logger.info("FAISS index directory: %s", settings.faiss_index_dir)

    yield

    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    """Construct and fully configure the FastAPI application.

    Using an explicit factory function (rather than a bare module-level
    `FastAPI()` call) keeps the app importable and re-constructible in
    tests — e.g. `create_app()` can be called once per test with
    `dependency_overrides` applied, without interference between tests.

    Returns:
        A fully configured `FastAPI` instance, ready to serve requests.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "A self-correcting, hybrid-retrieval Adaptive RAG system built "
            "entirely on free, open-source infrastructure."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # --- Middleware ----------------------------------------------------
    # Order matters: Starlette applies middleware in reverse of the order
    # added, so the LAST middleware added here is the FIRST to see the
    # request. We want RequestIDMiddleware to run first (so every
    # subsequent middleware/handler can rely on request.state.request_id
    # already being set), so it is added last.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # --- Exception handling ---------------------------------------------
    register_exception_handlers(app)

    # --- Routes ----------------------------------------------------------
    app.include_router(health_router, prefix=API_V1_PREFIX)
    app.include_router(chat_router, prefix=API_V1_PREFIX)
    app.include_router(upload_router, prefix=API_V1_PREFIX)
    app.include_router(documents_router, prefix=API_V1_PREFIX)
    app.include_router(reset_router, prefix=API_V1_PREFIX)
    app.include_router(metrics_router, prefix=API_V1_PREFIX)
    app.include_router(config_router, prefix=API_V1_PREFIX)

    # Unprefixed convenience alias so uptime monitors / container
    # orchestrators can hit /health without needing to know the API
    # version prefix.
    app.include_router(health_router)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        """Minimal landing payload pointing consumers at the API docs."""
        return {
            "app": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    _settings = get_settings()
    uvicorn.run(
        "backend.api.main:app",
        host=_settings.api_host,
        port=_settings.api_port,
        reload=_settings.debug,
        log_config=None,  # we manage logging ourselves via configure_logging
    )
