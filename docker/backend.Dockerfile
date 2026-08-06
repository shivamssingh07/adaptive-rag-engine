# ==============================================================================
# Adaptive RAG Engine — Backend (FastAPI) Docker image
#
# Build from repository root:
#   docker build -f docker/backend.Dockerfile -t adaptive-rag-backend .
#
# Local run:
#   docker run --env-file .env -p 8000:8000 adaptive-rag-backend
#
# Render:
#   Uses Render's PORT environment variable automatically.
# ==============================================================================

FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and buffering stdout/stderr.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System dependencies required by:
# - faiss-cpu
# - torch
# - sentence-transformers
# - PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application.
COPY backend/ backend/

# Create runtime directories.
RUN mkdir -p \
        data/uploads \
        data/faiss_index \
        data/bm25_index \
        logs

# Create non-root application user.
RUN groupadd --system app \
    && useradd --system \
        --gid app \
        --home-dir /app \
        app \
    && chown -R app:app /app

USER app

# Render provides the PORT environment variable.
# 8000 is the fallback for local Docker usage.
EXPOSE 8000

# Health check using the same PORT used by the application.
HEALTHCHECK --interval=30s \
    --timeout=5s \
    --start-period=30s \
    --retries=3 \
    CMD-SHELL \
    curl -f http://localhost:${PORT:-8000}/health || exit 1

# Start FastAPI.
# Render automatically provides PORT.
# Locally, if PORT is not defined, 8000 is used.
CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]