# Deployment Guide

## Local (see [`docs/installation.md`](installation.md))

## Docker Compose (recommended for self-hosting)

```bash
cp .env.example .env   # set GROQ_API_KEY
docker compose -f docker/docker-compose.yml up --build
```

Two services:
- `backend` — FastAPI, port `8000`, indexes/uploads persisted to `./data` via bind mount
- `frontend` — Streamlit, port `8501`, waits for the backend's healthcheck before starting

**Why two separate images instead of one combined container?** The backend needs `torch`/`faiss`/`langgraph` (large); the frontend needs only `streamlit`/`httpx` (small). Building them separately means the frontend image builds in seconds and stays under 300 MB, rather than inheriting the backend's multi-GB dependency footprint. It also mirrors how you'd actually scale this in production — the backend might need a GPU-enabled node pool; the frontend never would.

### Updating

```bash
docker compose -f docker/docker-compose.yml up --build
```

Rebuilds only the layers that changed (dependency layers are cached separately from application-code layers in both Dockerfiles).

### Logs

```bash
docker compose -f docker/docker-compose.yml logs -f backend
docker compose -f docker/docker-compose.yml logs -f frontend
```

## Render

1. Create a **Web Service** from this repo, root directory = repo root.
2. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn backend.api.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variable `GROQ_API_KEY` (and optionally `TAVILY_API_KEY`).
5. Add a **persistent disk** mounted at `/opt/render/project/src/data` so FAISS/BM25/SQLite survive redeploys.
6. Create a second Web Service for the frontend: build command `pip install -r docker/frontend-requirements.txt`, start command `streamlit run frontend/streamlit_app/app.py --server.address=0.0.0.0 --server.port=$PORT`, environment variable `API_BASE_URL` pointing at the backend service's public URL + `/api/v1`.

## Railway

Same two-service pattern as Render:
1. New project → deploy from repo → this becomes the **backend** service. Set start command to `uvicorn backend.api.main:app --host 0.0.0.0 --port $PORT` and add a volume mounted at `/app/data`.
2. Add a second service from the same repo for the **frontend**: start command `streamlit run frontend/streamlit_app/app.py --server.address=0.0.0.0 --server.port=$PORT`, with `API_BASE_URL` set to the backend service's internal Railway URL + `/api/v1`.
3. Set `GROQ_API_KEY` (and optionally `TAVILY_API_KEY`) on the backend service.

## Streamlit Cloud (frontend only)

Streamlit Cloud hosts the frontend only — you still need the backend deployed somewhere reachable (Render, Railway, your own server, etc.).

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at `frontend/streamlit_app/app.py`.
3. Under **Advanced settings → Secrets**, add:
   ```toml
   API_BASE_URL = "https://your-backend-url.example.com/api/v1"
   ```
   (Streamlit Cloud injects secrets as environment variables, which `api_client.py` reads via `API_BASE_URL`.)
4. Deploy. Note Streamlit Cloud's free tier only hosts the frontend — the backend must be reachable over the public internet from wherever it's deployed.

## Production configuration checklist

- [ ] `ENVIRONMENT=production` in `.env`
- [ ] `DEBUG=false`
- [ ] `CORS_ORIGINS` set to your actual frontend origin(s), not `*`
- [ ] `LOG_JSON=true` (structured logs for your log aggregator)
- [ ] A persistent volume/disk for `data/` (FAISS, BM25, SQLite) — without this, every redeploy wipes the knowledge base and conversation history
- [ ] `GROQ_API_KEY` and (optionally) `TAVILY_API_KEY` set as secrets, not committed to the repo
