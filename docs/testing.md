# Testing Guide

## Running the suite

```bash
pip install -r requirements-dev.txt
pytest
```

Runs 72 tests (unit + integration) with coverage reporting (configured in `pyproject.toml`'s `[tool.pytest.ini_options]`). For an HTML coverage report:

```bash
pytest --cov-report=html
open htmlcov/index.html
```

Run only unit tests (fast, no HTTP layer):
```bash
pytest tests/unit -v
```

Run only integration tests (full FastAPI + LangGraph, still no real network calls):
```bash
pytest tests/integration -v
```

## Why the suite never calls the real Groq API or downloads real models

Every class that touches an external model (`GroqLLMProvider`, `HuggingFaceEmbeddingProvider`, `CrossEncoderReranker`, and everything built on top of them) was designed from Phase 3 onward to accept its dependency through a constructor parameter — `FAISSVectorStore(settings, embeddings=...)`, `HybridRetriever(vector_retriever=..., bm25_retriever=...)`, and so on. Production code omits the parameter and gets the real lazy-loaded singleton; tests pass in a fake.

`tests/conftest.py` provides:
- **`FakeEmbeddings`** — a deterministic, hash-based embedding function. No model download, no network call, but real enough that FAISS's actual similarity search logic is genuinely exercised (similar text really does get more similar vectors).
- **`FakeRerankerModel`** / **`FakeGroqProvider`** (wrapping LangChain's own `FakeListChatModel`) — same idea, for reranking and LLM calls.
- **`api_client`** — a `TestClient` wired to fully isolated fakes via `app.dependency_overrides` (FastAPI's native DI override mechanism) for anything that goes through a route handler, plus direct monkeypatching of the module-level singletons the LangGraph nodes call directly (since graph nodes execute deep inside a route handler, not as route handlers themselves, and never go through FastAPI's DI).

This means the test suite genuinely exercises real FAISS index construction/search/persistence, real BM25 scoring, real SQLite reads/writes, real LangGraph compilation and execution (routing, conditional edges, retry loops), and real FastAPI request/response handling — the only thing that's fake is the actual neural network forward pass, which is exactly the part that would make the suite slow, non-deterministic, and dependent on paid API credits.

## Bugs the test suite actually caught

Worth knowing about, since they're a good illustration of why this approach works:

1. **`ScoredDocument` leaking `numpy.float32`** — FAISS/BM25 return numpy scalar types; left alone, this broke `json.dumps` the first time a citation reached an API response. Fixed by coercing to a native `float` in `ScoredDocument.__post_init__`.
2. **Five LLM call sites with a graceful-degradation bug** — `router.py`, `grade_documents.py`, `grade_generation.py`, `query_rewriter.py`, `multi_query_retriever.py`, and `self_query_retriever.py` all had `provider.get_llm(...)` called *outside* their `try/except` block, meaning a Groq client construction failure would crash instead of triggering the intended fallback behavior (default route, default grade, original query, etc.).
3. **`get_indexer()`/`get_conversation_memory()` bypassing FastAPI's own DI override system** — they called the raw singleton getters directly instead of depending on the already-overridable `get_faiss_store`/`get_bm25_index`/`get_document_registry`/`get_session_store` providers. This wasn't just a test-isolation inconvenience — it was a real production inconsistency in how the dependency-injection system was supposed to work, caught because upload → list-documents cross-request state didn't match in a test run.

None of these were found by manual code review; all were found by actually running the code.

## Test organization

```
tests/
├── conftest.py                     # Shared fixtures: fakes, settings, api_client
├── unit/
│   ├── test_settings.py            # Settings validation, computed properties
│   ├── test_loaders.py             # PDF/DOCX/TXT/MD/CSV parsing, real files
│   ├── test_splitter.py            # Chunking behavior
│   ├── test_hybrid_retriever.py    # BM25+vector fusion, score types
│   ├── test_reranker.py            # Cross-encoder wrapper logic (model mocked)
│   └── test_graph_nodes.py         # Individual LangGraph nodes in isolation
└── integration/
    ├── test_upload_endpoint.py     # Upload, list, delete, reset — via TestClient
    ├── test_chat_endpoint.py       # Chat (streaming + non-streaming), sessions
    └── test_full_graph_run.py      # The COMPILED graph, full routing/retry behavior
```

## Writing new tests

Reach for the fixtures in `conftest.py` rather than hand-rolling mocks:
- Testing a single RAG component? Use `settings`, `fake_embeddings`, `faiss_store`, `bm25_index`, `sample_documents`.
- Testing a graph node in isolation? Build a `GraphState` directly and call the node function; wire singletons via the `_wire_singletons`-style helper pattern in `test_graph_nodes.py`.
- Testing an API endpoint or the full graph end-to-end? Use `api_client` and `configure_llm_responses`.

One gotcha worth knowing: `AdaptiveRetriever` picks its strategy partly by query word count (`multi_query` for queries ≤ 6 words). When hand-crafting a canned LLM response sequence for a test, use queries and rewritten-query responses longer than 6 words unless you're specifically testing the `multi_query` path — otherwise an extra, unplanned LLM call shifts every subsequent canned response by one and the test fails in a confusing way. (This bit us during development — see the git history / this doc for the fix pattern.)
