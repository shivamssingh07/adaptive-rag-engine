# ==============================================================================
# Adaptive RAG Engine — Frontend (Streamlit) Docker image
#
# Build from the repository root:
#   docker build -f docker/frontend.Dockerfile -t adaptive-rag-frontend .
#
# Run:
#   docker run -e API_BASE_URL=http://backend:8000/api/v1 -p 8501:8501 adaptive-rag-frontend
# ==============================================================================

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# The frontend only needs Streamlit + httpx — installing the full
# requirements.txt would needlessly pull in torch/faiss/langgraph into an
# image that never uses them, roughly doubling build time and image size
# for no benefit.
COPY docker/frontend-requirements.txt .
RUN pip install --no-cache-dir -r frontend-requirements.txt

COPY frontend/ frontend/
COPY .streamlit/ .streamlit/

# Run as a non-root user — standard container security practice.
RUN groupadd --system app && useradd --system --gid app --home-dir /app app \
    && chown -R app:app /app
USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENV API_BASE_URL=http://backend:8000/api/v1

CMD ["streamlit", "run", "frontend/streamlit_app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
