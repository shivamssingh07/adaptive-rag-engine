# API Reference

Base URL: `http://localhost:8000/api/v1` (interactive Swagger UI at `/docs`, ReDoc at `/redoc`).

Every error response uses the same envelope, regardless of endpoint:

```json
{
  "error_code": "document_not_found",
  "message": "Document 'doc_abc123' does not exist.",
  "details": {"document_id": "doc_abc123"},
  "request_id": "req_...",
  "timestamp": "2026-01-01T12:00:00Z"
}
```

---

## `POST /chat`

Ask a question. Runs the full adaptive RAG graph for one conversation turn.

**Request:**
```json
{
  "message": "What is the refund policy?",
  "session_id": null,
  "stream": true
}
```

- `message` (required, 1–4000 chars)
- `session_id` (optional) — omit to start a new conversation
- `stream` (default `true`) — SSE if true, single JSON body if false

**Streaming response** (`stream: true`, `Content-Type: text/event-stream`):
```
event: token
data: {"content": "Refunds"}

event: token
data: {"content": " are"}

...

event: done
data: {"session_id": "sess_...", "sources": [...], "metrics": {...}}
```

**Non-streaming response** (`stream: false`):
```json
{
  "session_id": "sess_a1b2c3...",
  "answer": "Refunds are accepted within 30 days of purchase.",
  "sources": [
    {
      "source": "policy.pdf",
      "document_id": "doc_...",
      "chunk_id": "doc_..._chunk_0",
      "page": 3,
      "row": null,
      "file_type": "pdf",
      "score": 0.91,
      "excerpt": "Refunds are processed within 30 days..."
    }
  ],
  "metrics": {
    "latency_ms": 842.3,
    "route": "vectorstore",
    "retrieval_strategy": "hybrid",
    "used_web_search": false,
    "document_relevance_retries": 0,
    "groundedness_retries": 0,
    "context_chunks_used": 3,
    "token_usage": {"input_tokens": 412, "output_tokens": 38, "total_tokens": 450}
  }
}
```

## `GET /chat/{session_id}/export`

Returns the full conversation transcript as plain text.

## `DELETE /chat/{session_id}`

Clears a session's message history (the session ID itself remains valid for continued use).

## `POST /upload`

Multipart form upload, one or more files under the `files` field.

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "files=@report.pdf" -F "files=@notes.txt"
```

**Response:**
```json
{
  "total_files": 2,
  "successful_files": 2,
  "failed_files": 0,
  "total_chunks_added": 14,
  "results": [
    {"filename": "report.pdf", "success": true, "chunks_added": 11, "document_id": "doc_...", "duplicate": false, "error": null},
    {"filename": "notes.txt", "success": true, "chunks_added": 3, "document_id": "doc_...", "duplicate": false, "error": null}
  ]
}
```

Failures are per-file — one corrupted or unsupported file never blocks the rest of the batch. Re-uploading identical content (by SHA-256 hash) returns `"duplicate": true, "chunks_added": 0` without re-indexing.

## `GET /documents`

```json
{
  "total_documents": 2,
  "total_chunks": 14,
  "documents": [
    {"document_id": "doc_...", "filename": "report.pdf", "file_type": "pdf", "chunk_count": 11, "size_bytes": 84213, "uploaded_at": "2026-01-01T12:00:00Z"}
  ]
}
```

## `DELETE /documents/{document_id}`

Removes every chunk of that document from both FAISS and BM25, plus its registry entry. Returns `404` (`document_not_found`) if the ID doesn't exist.

## `POST /reset`

Clears the entire knowledge base (all documents, both indexes). Irreversible — no confirmation at the API layer; the Streamlit UI confirms with the user before calling this.

## `GET /health`

```json
{
  "status": "healthy",
  "app_name": "Adaptive RAG Engine",
  "version": "1.0.0",
  "environment": "development",
  "uptime_seconds": 1234.5,
  "components": [
    {"name": "groq_llm", "configured": true, "detail": "model=llama-3.3-70b-versatile"},
    {"name": "tavily_web_search", "configured": false, "detail": "optional fallback; disabled when TAVILY_API_KEY is unset"}
  ]
}
```

## `GET /metrics`

```json
{
  "total_documents": 2,
  "total_chunks_faiss": 14,
  "total_chunks_bm25": 14,
  "active_sessions": 3,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "reranker_model": "BAAI/bge-reranker-base",
  "llm_model": "llama-3.3-70b-versatile",
  "tavily_web_search_enabled": false
}
```

## `GET /config`

Non-sensitive current configuration (never includes API keys) — see `docs/architecture.md#configuration-reference` for every field.
