"""FastAPI dependency-injection providers.

Centralizing dependency providers here (rather than importing singletons
directly into route modules) keeps route handlers testable: tests can
override any of these with `app.dependency_overrides[...]` without
monkeypatching module-level globals.
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, Request
from langgraph.graph.state import CompiledStateGraph

from backend.config.settings import Settings
from backend.config.settings import get_settings as _get_settings
from backend.core.graph.builder import get_compiled_graph as _get_compiled_graph
from backend.core.graph.state import GraphState
from backend.rag.indexing.bm25_index import BM25Index
from backend.rag.indexing.bm25_index import get_bm25_index as _get_bm25_index
from backend.rag.indexing.document_registry import (
    DocumentRegistry,
)
from backend.rag.indexing.document_registry import (
    get_document_registry as _get_document_registry,
)
from backend.rag.indexing.faiss_store import FAISSVectorStore
from backend.rag.indexing.faiss_store import get_faiss_store as _get_faiss_store
from backend.rag.indexing.indexer import Indexer
from backend.rag.memory.conversation_memory import ConversationMemory
from backend.rag.memory.session_store import SessionStore
from backend.rag.memory.session_store import get_session_store as _get_session_store


def get_settings() -> Settings:
    """Dependency provider for the application settings singleton."""
    return _get_settings()


def get_request_id(request: Request) -> str:
    """Dependency provider for the current request's correlation ID.

    The ID is bound onto `request.state` by
    `backend.api.middleware.request_id.RequestIDMiddleware` earlier in the
    middleware chain; this provider simply reads it back out for handlers
    and other dependencies that need to include it in responses or logs.
    """
    return getattr(request.state, "request_id", "unknown")


def get_app_start_time(request: Request) -> float:
    """Dependency provider for the process start time, used to compute
    uptime in the health endpoint."""
    return float(request.app.state.start_time)


def get_faiss_store() -> FAISSVectorStore:
    """Dependency provider for the process-wide FAISS vector store."""
    return _get_faiss_store()


def get_bm25_index() -> BM25Index:
    """Dependency provider for the process-wide BM25 lexical index."""
    return _get_bm25_index()


def get_document_registry() -> DocumentRegistry:
    """Dependency provider for the process-wide document registry."""
    return _get_document_registry()


def get_session_store() -> SessionStore:
    """Dependency provider for the process-wide session store."""
    return _get_session_store()


def get_conversation_memory(
    session_store: SessionStore = Depends(get_session_store),
) -> ConversationMemory:
    """Dependency provider for a `ConversationMemory`, composed from the
    already-declared `get_session_store` dependency so overriding it in
    tests flows through here too."""
    return ConversationMemory(session_store=session_store)


def get_indexer(
    faiss_store: FAISSVectorStore = Depends(get_faiss_store),
    bm25_index: BM25Index = Depends(get_bm25_index),
    document_registry: DocumentRegistry = Depends(get_document_registry),
) -> Indexer:
    """Dependency provider for an `Indexer`, composed from the
    already-declared `get_faiss_store`/`get_bm25_index`/`get_document_registry`
    dependencies (rather than calling their underlying singleton getters
    directly) so that overriding those three via
    `app.dependency_overrides` in tests automatically flows through to
    every route that depends on `get_indexer`, too.
    """
    return Indexer(
        faiss_store=faiss_store,
        bm25_index=bm25_index,
        document_registry=document_registry,
    )


def get_graph() -> CompiledStateGraph[GraphState, Any, Any, Any]:
    """Dependency provider for the compiled adaptive RAG LangGraph."""
    return _get_compiled_graph()
