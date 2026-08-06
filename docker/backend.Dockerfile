# ==============================================================================
# Adaptive RAG Engine — Backend (FastAPI) Docker image
#
# Build from the repository root:
#   docker build -f docker/backend.Dockerfile -t adaptive-rag-backend .
#
# Run:
#   docker run --env-file .env -p 8000:8000 adaptive-rag-backend
# ==============================================================================

FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and buffering stdout/stderr, so
# `docker logs` shows output in real time.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System dependencies required to build/run faiss-cpu, torch,
# sentence-transformers, and PyMuPDF's PDF parsing.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so this layer is cached independently
# of application code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY backend/ backend/

# Runtime data directories (mounted as volumes in docker-compose, but
# created here too so the image is self-sufficient when run standalone).
RUN mkdir -p data/uploads data/faiss_index data/bm25_index logs

# Run as a non-root user — standard container security practice. The data/
# and logs/ directories must be writable by this user (they're also the
# mount points for docker-compose's volumes, which inherit host ownership;
# see README's Docker troubleshooting section if a bind-mounted volume
# ends up owned by root on the host).
RUN groupadd --system app && useradd --system --gid app --home-dir /app app \
    && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
