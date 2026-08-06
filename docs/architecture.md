# Architecture

## System overview

```mermaid
flowchart TB
    subgraph Client["Client Layer"]
        UI[Streamlit Dashboard]
    end

    subgraph API["Backend Layer — FastAPI"]
        R1["/chat (SSE streaming)"]
        R2["/upload"]
        R3["/health"]
        R4["/reset"]
        R5["/documents"]
        R6["/metrics, /config"]
    end

    subgraph Core["Core Orchestration — LangGraph"]
        G[Adaptive RAG Graph]
        RT[Router Node]
        RV[Retrieve Node]
        GR[Grade Documents Node]
        RW[Rewrite Query Node]
        WS[Web Search Node]
        GN[Generate Node]
        GG[Grade Generation Node]
    end

    subgraph RAGModules["RAG Subsystems"]
        EMB[Embeddings — MiniLM]
        VS[FAISS Vector Store]
        BM[BM25 Index]
        RR[Cross-Encoder Reranker]
        CC[Contextual Compressor]
        MEM[Conversation Memory — SQLite]
        REG[Document Registry — SQLite]
    end

    UI -->|HTTP/SSE| R1
    UI --> R2
    UI --> R3
    UI --> R4
    UI --> R5
    UI --> R6

    R2 --> REG
    R2 --> VS
    R2 --> BM

    R1 --> G
    G --> RT --> RV
    RT -.no docs / web route.-> WS
    RV --> VS
    RV --> BM
    RV --> RR
    RV --> CC
    RV --> GR
    GR -->|relevant| GN
    GR -->|irrelevant, retries left| RW --> RV
    GR -->|retries exhausted, web enabled| WS --> GN
    GN --> GG
    GG -->|grounded| R1
    GG -->|not grounded, retries left| GN
    G <--> MEM
```

## The adaptive RAG graph (LangGraph state machine)

```mermaid
flowchart LR
    START([User Query]) --> A[route_question]
    A -->|vectorstore| B[retrieve]
    A -->|web_search| F[web_search]
    A -->|direct_answer| D[generate]

    B --> C[grade_documents]
    C -->|relevant| D
    C -->|irrelevant, retries left| E[rewrite_query]
    C -->|irrelevant, exhausted, web enabled| F
    C -->|irrelevant, exhausted, no web| D

    E --> B
    F --> D

    D --> H[grade_generation]
    H -->|grounded| END1([Return Answer])
    H -->|not grounded, retries left| D
    H -->|not grounded, exhausted| END1
```

Implementation: `backend/core/graph/builder.py` wires this exact graph; each node is a thin wrapper in `backend/core/graph/nodes/` around the RAG primitives in `backend/rag/`.

**Retry bounds** (both configurable, default 2): `MAX_DOCUMENT_GRADE_RETRIES`, `MAX_GROUNDEDNESS_RETRIES`. The graph always terminates — it never loops forever, even if the LLM keeps grading content as irrelevant/ungrounded.

## Sequence: one `/chat` request

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit UI
    participant API as FastAPI /chat
    participant Graph as LangGraph Runtime
    participant Retr as Adaptive Retriever
    participant Rerank as Cross-Encoder
    participant LLM as Groq Llama-3.3-70B
    participant Mem as SQLite Session Store

    User->>UI: Types question
    UI->>API: POST /chat (session_id, message)
    API->>Mem: Load conversation history
    API->>Graph: graph.invoke(state)  [run to completion, off the event loop]
    Graph->>LLM: route_question
    Graph->>Retr: hybrid / mmr / multi_query / self_query
    Retr-->>Graph: candidate chunks
    Graph->>Rerank: rerank(query, candidates)
    Rerank-->>Graph: top-k reranked
    Graph->>LLM: grade_documents
    alt relevant
        Graph->>LLM: generate
        Graph->>LLM: grade_generation
    else irrelevant (bounded retries)
        Graph->>LLM: rewrite_query
        Graph->>Retr: retry retrieval
    end
    API->>Mem: persist turn (with citations)
    API-->>UI: SSE stream (verified answer, then citations+metrics)
```

**Why streaming happens after the graph completes, not token-by-token through it**: the self-correction loop may discard and regenerate the answer if it fails the groundedness check. You cannot un-stream tokens the client has already rendered. The API layer therefore runs the graph to completion (off the event loop, via `run_in_threadpool`), then streams the *final, verified* answer to the client in chunks — preserving a real-time UX without ever showing the user an answer that later gets thrown away.

## Ingestion pipeline

```mermaid
flowchart TD
    U[User uploads file] --> V{Validate extension & size}
    V -->|invalid| E1[Return per-file error, continue batch]
    V -->|valid| H{SHA-256 hash already registered?}
    H -->|yes| SKIP[Skip — mark duplicate, no re-indexing]
    H -->|no| P{Route by extension}
    P -->|.pdf| L1[PyMuPDF]
    P -->|.docx| L2[python-docx]
    P -->|.txt/.md| L3[Plain text]
    P -->|.csv| L4[pandas]
    L1 & L2 & L3 & L4 --> N{Parse succeeded?}
    N -->|no| E2[Log, return per-file error, continue batch]
    N -->|yes| S[RecursiveCharacterTextSplitter]
    S --> EMB[Embed chunks — MiniLM]
    EMB --> FW[(FAISS)]
    S --> BW[(BM25)]
    FW & BW --> REG[(Document Registry)]
    REG --> DONE[Return batch summary]
```

## Module dependency graph

Dependencies flow strictly one direction: `config → rag primitives → retrievers → graph nodes → graph builder → API → frontend`. No circular imports — verified in CI via a full `pkgutil.walk_packages` import audit.

```mermaid
flowchart TD
    settings[config.settings] --> llm[rag.llms.groq_provider]
    settings --> emb[rag.embeddings.huggingface_provider]
    settings --> faiss_store[rag.indexing.faiss_store]
    settings --> tavily[rag.search.tavily_search]

    emb --> faiss_store
    faiss_store --> vector_retriever[rag.retrievers.vector_retriever]
    bm25_index[rag.indexing.bm25_index] --> bm25_retriever[rag.retrievers.bm25_retriever]

    loaders[rag.loaders.*] --> indexer[rag.indexing.indexer]
    faiss_store --> indexer
    bm25_index --> indexer
    registry[rag.indexing.document_registry] --> indexer

    vector_retriever --> hybrid[rag.retrievers.hybrid_retriever]
    bm25_retriever --> hybrid
    hybrid --> adaptive[rag.retrievers.adaptive_retriever]
    llm --> adaptive

    adaptive --> nodes[core.graph.nodes.*]
    llm --> nodes
    tavily --> nodes

    nodes --> builder[core.graph.builder]
    builder --> api_deps[api.dependencies]
    indexer --> api_deps
    api_deps --> routes[api.routes.*]
    routes --> main[api.main]

    main --> client[frontend.streamlit_app.services.api_client]
    client --> app[frontend.streamlit_app.app]
```

## Key design decisions

| Decision | Rationale |
|---|---|
| Pydantic `GraphState` instead of `TypedDict` | Runtime validation of node return shapes catches bugs at dev time. |
| Lazy-loaded singletons for LLM/embeddings/reranker | Expensive model/client construction is deferred to first use and cached — fast cold starts, fast tests. |
| Hybrid retrieval (BM25 + vector) as the default strategy | Vector search alone misses exact keyword/entity matches; BM25 alone misses semantic paraphrase. |
| Bounded retry loops (not unbounded) | Guarantees termination even for genuinely unanswerable questions. |
| Tavily fully optional, checked once via `settings.tavily_enabled` | Keeps the zero-paid-dependency promise true for every code path, not just the happy path. |
| Separate backend/frontend processes over HTTP (not a monolith) | Independently deployable/scalable; the frontend never imports `backend.*`. |
| SQLite for sessions and the document registry (not in-memory dicts) | Conversation history and document metadata survive a process restart, at zero infra cost. |
| Structured JSON logging with request-ID correlation | Every log line from router → retrieval → grading → generation for one request is joinable. |
| `get_indexer`/`get_conversation_memory` compose FastAPI `Depends(...)` rather than calling singleton getters directly | Preserves `app.dependency_overrides` testability end-to-end — this was a real bug caught by the test suite (see `docs/testing.md`). |

## Configuration reference

All settings live in `backend/config/settings.py`, sourced from environment variables / `.env`. Only `GROQ_API_KEY` is required.

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API credential |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | LLM model name |
| `GROQ_TEMPERATURE` | `0.1` | Default sampling temperature |
| `TAVILY_API_KEY` | *(unset)* | Enables web-search fallback if set |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-base` | Cross-encoder reranker model |
| `FAISS_INDEX_DIR` | `data/faiss_index` | FAISS persistence directory |
| `BM25_INDEX_DIR` | `data/bm25_index` | BM25 persistence directory |
| `UPLOAD_DIR` | `data/uploads` | Raw uploaded file storage |
| `SESSION_DB_PATH` | `data/sessions.db` | Conversation history SQLite DB |
| `DOCUMENT_REGISTRY_DB_PATH` | `data/documents.db` | Document metadata SQLite DB |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Text splitter configuration |
| `MAX_UPLOAD_SIZE_MB` | `25` | Per-file upload size cap |
| `ALLOWED_UPLOAD_EXTENSIONS` | `.pdf,.docx,.txt,.md,.csv` | Accepted upload formats |
| `TOP_K_RETRIEVAL` / `TOP_K_RERANK` | `10` / `4` | Retrieval and reranking depth |
| `HYBRID_BM25_WEIGHT` | `0.4` | BM25 weight in hybrid fusion (vector weight = `1 - this`) |
| `MAX_DOCUMENT_GRADE_RETRIES` | `2` | Bounded query-rewrite retry limit |
| `MAX_GROUNDEDNESS_RETRIES` | `2` | Bounded regeneration retry limit |
| `LOG_LEVEL` / `LOG_JSON` | `INFO` / `true` | Logging verbosity and format |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | Backend bind address |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
