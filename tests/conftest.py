"""Pytest configuration and shared fixtures for the Adaptive RAG Engine
test suite.

Design notes:
    * `langchain_huggingface`/`sentence_transformers` are stubbed ONLY if
      genuinely not importable (e.g. a CI/dev sandbox that cannot fit the
      `torch` wheel on disk). In a real environment with the full
      `requirements.txt` installed, these stubs never activate — the real
      packages are used, but every fixture in this file injects a fake
      embeddings/reranker/LLM object directly into the constructor
      parameters those classes were explicitly designed to accept for
      testability, so no test ever actually needs the real model weights.
    * A minimal `GROQ_API_KEY` is set at MODULE import time (not inside a
      fixture) because `backend.api.main` constructs its FastAPI `app`
      object at import time via `get_settings()`, which happens before any
      function-scoped fixture would run. Per-test isolation for the RAG
      subsystems (FAISS/BM25/sessions/registry) is instead layered on top
      via `app.dependency_overrides` — see the `api_client` fixture.
"""

from __future__ import annotations

import hashlib
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

# --------------------------------------------------------------------------
# Stub torch-dependent packages, ONLY if genuinely not installed.
# --------------------------------------------------------------------------
try:
    import langchain_huggingface  # noqa: F401
except ImportError:
    _stub_hf = types.ModuleType("langchain_huggingface")

    class _UnavailableHuggingFaceEmbeddings:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "langchain_huggingface is not installed in this test environment. "
                "Tests should never construct it directly — inject the "
                "`fake_embeddings` fixture instead."
            )

    _stub_hf.HuggingFaceEmbeddings = _UnavailableHuggingFaceEmbeddings  # type: ignore[attr-defined]
    sys.modules["langchain_huggingface"] = _stub_hf

try:
    import sentence_transformers  # noqa: F401
except ImportError:
    _stub_st = types.ModuleType("sentence_transformers")

    class _UnavailableCrossEncoder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "sentence_transformers is not installed in this test environment. "
                "Tests should never construct it directly — inject the "
                "`fake_reranker` fixture instead."
            )

    _stub_st.CrossEncoder = _UnavailableCrossEncoder  # type: ignore[attr-defined]
    sys.modules["sentence_transformers"] = _stub_st

# --------------------------------------------------------------------------
# Minimal env bootstrap so `backend.api.main`'s module-level `app = create_app()`
# succeeds the first time any test module imports it.
# --------------------------------------------------------------------------
os.environ.setdefault("GROQ_API_KEY", "test_dummy_key_for_pytest")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("LOG_JSON", "false")

from backend.config.settings import Settings, get_settings  # noqa: E402


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """A fresh, fully isolated `Settings` instance backed by a per-test
    temporary directory.

    Use this when constructing RAG components directly in unit tests —
    every one of them accepts an explicit `settings` parameter, bypassing
    the global `lru_cache`-d singleton entirely.
    """
    monkeypatch.setenv("GROQ_API_KEY", "test_dummy_key_for_pytest")
    monkeypatch.setenv("FAISS_INDEX_DIR", str(tmp_path / "faiss_index"))
    monkeypatch.setenv("BM25_INDEX_DIR", str(tmp_path / "bm25_index"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("SESSION_DB_PATH", str(tmp_path / "sessions.db"))
    monkeypatch.setenv("DOCUMENT_REGISTRY_DB_PATH", str(tmp_path / "documents.db"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("TAVILY_API_KEY", "")
    get_settings.cache_clear()
    fresh_settings = get_settings()
    yield fresh_settings
    get_settings.cache_clear()


# --------------------------------------------------------------------------
# Fakes: embeddings, reranker, LLM
# --------------------------------------------------------------------------
class FakeEmbeddings(Embeddings):
    """Deterministic, hash-based fake embedding model requiring no real
    model weights. Similar text hashes to similar (not identical) vectors
    — sufficient to exercise real FAISS similarity search logic without a
    network call or a multi-hundred-MB model download."""

    _DIM = 32

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self._DIM, dtype=np.float32)
        for token in text.lower().split():
            digest = int(hashlib.md5(token.encode()).hexdigest(), 16)  # noqa: S324
            vector[digest % self._DIM] += 1.0
        norm = np.linalg.norm(vector)
        return (vector / norm if norm > 0 else vector).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


class FakeRerankerModel:
    """Drop-in replacement for `CrossEncoderReranker`: preserves input
    order and assigns descending synthetic scores."""

    def rerank(
        self, query: str, documents: list[Document], top_k: int | None = None
    ) -> list[tuple[Document, float]]:
        k = top_k if top_k is not None else len(documents)
        return [(doc, 1.0 - index * 0.01) for index, doc in enumerate(documents[:k])]


@pytest.fixture
def fake_reranker() -> FakeRerankerModel:
    return FakeRerankerModel()


class FakeGroqProvider:
    """Drop-in replacement for `GroqLLMProvider`: returns a
    `FakeListChatModel` cycling through a supplied list of canned
    responses."""

    def __init__(self, responses: list[str]) -> None:
        self._llm = FakeListChatModel(responses=responses)

    def get_client(self) -> FakeListChatModel:
        return self._llm

    def get_llm(self, temperature: float | None = None) -> FakeListChatModel:
        return self._llm


@pytest.fixture
def make_fake_groq_provider():
    """Factory fixture: call with a list of canned responses to get a
    `FakeGroqProvider`."""

    def _factory(responses: list[str]) -> FakeGroqProvider:
        return FakeGroqProvider(responses=responses)

    return _factory


# --------------------------------------------------------------------------
# Sample data
# --------------------------------------------------------------------------
@pytest.fixture
def sample_documents() -> list[Document]:
    """A small, fixed corpus of chunks used across many unit tests."""
    return [
        Document(
            page_content=(
                "The refund policy allows returns within 30 days of purchase for a full refund."
            ),
            metadata={
                "source": "policy.pdf",
                "document_id": "doc_1",
                "page": 1,
                "chunk_id": "doc_1_chunk_0",
                "file_type": "pdf",
            },
        ),
        Document(
            page_content="International orders are eligible for store credit only, not cash refunds.",
            metadata={
                "source": "policy.pdf",
                "document_id": "doc_1",
                "page": 2,
                "chunk_id": "doc_1_chunk_1",
                "file_type": "pdf",
            },
        ),
        Document(
            page_content="The cafeteria menu includes vegetarian options every Tuesday and Thursday.",
            metadata={
                "source": "notes.txt",
                "document_id": "doc_2",
                "chunk_id": "doc_2_chunk_0",
                "file_type": "text",
            },
        ),
    ]


# --------------------------------------------------------------------------
# RAG subsystem fixtures (unit-test level)
# --------------------------------------------------------------------------
@pytest.fixture
def faiss_store(settings: Settings, fake_embeddings: FakeEmbeddings):
    from backend.rag.indexing.faiss_store import FAISSVectorStore

    return FAISSVectorStore(settings=settings, embeddings=fake_embeddings)


@pytest.fixture
def bm25_index(settings: Settings):
    from backend.rag.indexing.bm25_index import BM25Index

    return BM25Index(settings=settings)


@pytest.fixture
def document_registry(settings: Settings):
    from backend.rag.indexing.document_registry import DocumentRegistry

    return DocumentRegistry(settings=settings)


@pytest.fixture
def indexer(settings: Settings, faiss_store, bm25_index, document_registry):
    from backend.rag.indexing.indexer import Indexer

    return Indexer(
        settings=settings,
        faiss_store=faiss_store,
        bm25_index=bm25_index,
        document_registry=document_registry,
    )


# --------------------------------------------------------------------------
# API-level fixture (integration-test level)
# --------------------------------------------------------------------------
@pytest.fixture
def api_client(
    settings: Settings,
    fake_embeddings: FakeEmbeddings,
    fake_reranker: FakeRerankerModel,
    make_fake_groq_provider,
):
    """A `TestClient` wired to fully isolated, fake-backed RAG subsystems.

    Each test gets its own empty FAISS/BM25/session/document-registry
    state (via `app.dependency_overrides`, FastAPI's native DI override
    mechanism) and deterministic, network-free LLM responses (via
    monkeypatched module-level singletons, since the LangGraph nodes call
    `get_faiss_store()`/`get_groq_provider()`/etc. directly rather than
    through FastAPI's dependency injection).

    Yields the `TestClient` inside a `with` block so the app's lifespan
    (startup/shutdown) events run.
    """
    import backend.rag.indexing.bm25_index as bm25_module
    import backend.rag.indexing.faiss_store as faiss_module
    import backend.rag.llms.groq_provider as groq_module
    import backend.rag.rerankers.cross_encoder_reranker as reranker_module
    from backend.api.dependencies import (
        get_bm25_index,
        get_document_registry,
        get_faiss_store,
        get_session_store,
    )
    from backend.api.dependencies import get_settings as dep_get_settings
    from backend.api.main import app
    from backend.rag.indexing.bm25_index import BM25Index
    from backend.rag.indexing.document_registry import DocumentRegistry
    from backend.rag.indexing.faiss_store import FAISSVectorStore
    from backend.rag.memory.session_store import SessionStore

    test_faiss_store = FAISSVectorStore(settings=settings, embeddings=fake_embeddings)
    test_bm25_index = BM25Index(settings=settings)
    test_document_registry = DocumentRegistry(settings=settings)
    test_session_store = SessionStore(settings=settings)

    faiss_module._store_singleton = test_faiss_store
    bm25_module._index_singleton = test_bm25_index
    reranker_module._reranker_singleton = fake_reranker
    groq_module._provider_singleton = make_fake_groq_provider(["vectorstore", "yes", "yes"])

    app.dependency_overrides[get_faiss_store] = lambda: test_faiss_store
    app.dependency_overrides[get_bm25_index] = lambda: test_bm25_index
    app.dependency_overrides[get_document_registry] = lambda: test_document_registry
    app.dependency_overrides[get_session_store] = lambda: test_session_store
    app.dependency_overrides[dep_get_settings] = lambda: settings

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()
    faiss_module._store_singleton = None
    bm25_module._index_singleton = None
    reranker_module._reranker_singleton = None
    groq_module._provider_singleton = None


@pytest.fixture
def configure_llm_responses(make_fake_groq_provider):
    """Helper fixture: call with a list of canned responses to reconfigure
    the graph's LLM provider mid-test (after `api_client` already set a
    default sequence)."""
    import backend.rag.llms.groq_provider as groq_module

    def _configure(responses: list[str]) -> None:
        groq_module._provider_singleton = make_fake_groq_provider(responses)

    return _configure
