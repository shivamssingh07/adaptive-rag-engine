# Roadmap

Honest list of known limitations and planned improvements — not a marketing wishlist.

## Near-term

- **Concurrent contextual compression** — `ContextualCompressor.compress()` currently makes one LLM call per candidate document, sequentially. For the small (top-k) candidate sets this runs over it's acceptable, but batching these calls concurrently (`asyncio.gather` over async LLM calls) would meaningfully cut `/chat` latency.
- **True token-level streaming through the self-correction loop** — today, the graph runs to full completion before the verified answer is streamed to the client (see `docs/architecture.md` for why). A middle ground worth exploring: stream tentative tokens optimistically, then send a correction event if the groundedness grader rejects the answer and a regeneration occurs.
- **BM25 incremental indexing** — `BM25Okapi` has no native incremental-update API, so `BM25Index` rebuilds the whole index from the in-memory corpus on every write. Fine at the scale this project targets (thousands of chunks); would need to move to a proper inverted-index engine (OpenSearch/Elasticsearch) at meaningfully larger scale.

## Medium-term

- **Additional vector store backends** — an interface-compatible Qdrant or Chroma backend behind the same `FAISSVectorStore`-shaped interface, selectable via `.env`, for users who want a server-based vector store instead of local FAISS files.
- **Structured self-query filtering beyond `source`** — currently `SelfQueryRetriever` only extracts a source-filename filter. Extending to date ranges, file types, or custom metadata tags would need a small schema-description addition to the self-query prompt plus corresponding FAISS filter support.
- **Multi-user auth** — sessions today are identified by an opaque ID the client holds; there's no user-account layer. Adding one would mainly touch `SessionStore`'s schema and a new auth dependency in `backend/api/dependencies.py`.

## Long-term / exploratory

- **Evaluation harness** — a small labeled QA set + a script that runs it through the graph and reports retrieval precision/recall and answer groundedness over time, to catch regressions from prompt or retrieval-strategy changes.
- **Alternative LLM providers** — the `LLMProvider` protocol in `backend/rag/llms/base.py` was designed so a second provider (e.g. a local Ollama model) could be added without touching any node code, only a new provider module and a settings toggle.
- **Real-time collaborative sessions** — multiple users viewing/contributing to the same conversation, which would need a pub/sub layer beyond SQLite polling.
