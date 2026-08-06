# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses date-based milestones rather than semantic version tags, since it is an actively-developed portfolio project.

## [Unreleased]

### Added
- Full project audit tooling references in `docs/testing.md`.

## [1.0.0] — Initial complete release

### Added
- **Backend foundation**: `pydantic-settings`-based configuration, structured JSON logging with request-ID correlation, FastAPI app factory with lifespan management, global exception handling, domain exception hierarchy.
- **RAG engine**: Groq LLM provider, HuggingFace embeddings provider, FAISS vector store (persistent, auto-reloading), BM25 lexical index, multi-format document loaders (PDF/DOCX/TXT/Markdown/CSV), recursive text splitter, hybrid/MMR/multi-query/self-query retrievers, cross-encoder reranker, contextual compression, adaptive retrieval strategy selector, SQLite-backed conversation memory, source citation builder.
- **LangGraph adaptive RAG orchestration**: router, retrieve, grade-documents, rewrite-query, web-search (optional Tavily fallback), generate, grade-generation nodes wired into a corrective-RAG state machine with bounded self-correction retry loops.
- **Document management**: SHA-256 duplicate detection, per-document deletion from both indexes, document registry.
- **FastAPI integration layer**: `/chat` (SSE streaming), `/upload`, `/documents` (list/delete), `/reset`, `/health`, `/metrics`, `/config`, plus session export/clear endpoints.
- **Streamlit frontend**: chat page with streaming responses, drag-and-drop multi-file upload, source citations, per-turn metrics, document management page, read-only settings page, about page, dark theme.
- **Test suite**: 72 tests (unit + integration) using fakes injected at existing constructor seams — no real model downloads or API calls required to run the suite.
- **Docker**: separate backend/frontend images, `docker-compose.yml` with healthchecks.
- **Documentation**: architecture (with Mermaid diagrams), API reference, installation, deployment, testing, and roadmap guides.

### Fixed (found via testing, not manual review)
- `ScoredDocument` coerces scores to native `float`, preventing a `numpy.float32` JSON-serialization crash.
- Six LLM call sites (`router`, `grade_documents`, `grade_generation`, `query_rewriter`, `multi_query_retriever`, `self_query_retriever`, `contextual_compression`) had provider/client construction outside their `try/except` blocks, defeating their intended graceful-degradation behavior on a Groq client failure.
- `get_indexer()`/`get_conversation_memory()` FastAPI dependency providers were calling raw singleton getters directly instead of composing already-declared, overridable dependencies — breaking `app.dependency_overrides`-based test isolation and, more importantly, being an inconsistent use of the DI pattern in production code.
- Streamlit's script runner does not add the project root to `sys.path`, which would have broken every `from frontend.streamlit_app...` import; fixed with an explicit `sys.path` bootstrap at the top of `app.py`.
