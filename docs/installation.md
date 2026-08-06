# Installation Guide

## Prerequisites

- Python 3.12 or newer
- A free [Groq API key](https://console.groq.com/keys) (required)
- Optionally, a free [Tavily API key](https://tavily.com) (enables web-search fallback)
- ~3–5 GB free disk space (mostly for `torch`, used locally for embeddings/reranking — this is what makes those free instead of paid API calls)

## Step-by-step

### 1. Clone and enter the project

```bash
git clone <this-repo-url>
cd adaptive-rag-engine
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs the backend (FastAPI, LangChain/LangGraph, FAISS, sentence-transformers, PyMuPDF, etc.) and the frontend (Streamlit) into the same environment. For development (tests, linting, type-checking):

```bash
pip install -r requirements-dev.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

That's the only required value. Everything else has a working default — see [`docs/architecture.md`](architecture.md#configuration-reference) for the full list.

### 5. Run it

Two terminals:

```bash
# Terminal 1
uvicorn backend.api.main:app --reload

# Terminal 2
streamlit run frontend/streamlit_app/app.py
```

Open **http://localhost:8501**.

### 6. Verify it's working

```bash
curl http://localhost:8000/api/v1/health
```

Should return `"status": "healthy"`. If it says `"degraded"`, double-check `GROQ_API_KEY` in `.env`.

## First run: model downloads

On the **first** chat request or document upload, the backend downloads:
- `sentence-transformers/all-MiniLM-L6-v2` (~90 MB) — for embeddings
- `BAAI/bge-reranker-base` (~1.1 GB) — for reranking

Both are cached locally afterward (standard HuggingFace cache, usually `~/.cache/huggingface`) — no repeated downloads on subsequent runs.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ConfigurationError` on startup | `GROQ_API_KEY` missing or empty in `.env` |
| `pip install` runs out of disk space | `torch` is large; free up space or use a machine with more disk |
| Frontend shows "Cannot reach the backend API" | Backend isn't running, or `API_BASE_URL` (frontend env var) doesn't match where the backend is listening |
| First chat message is slow | Expected — first-use model downloads happen then; subsequent requests are fast |
| `ModuleNotFoundError: No module named 'frontend'` | You're running `streamlit run` from somewhere other than the project root, or bypassing the `sys.path` bootstrap at the top of `frontend/streamlit_app/app.py` |

See also [`docs/deployment.md`](deployment.md) for Docker-specific setup.
