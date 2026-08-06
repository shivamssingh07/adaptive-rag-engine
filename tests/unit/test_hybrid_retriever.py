"""Unit tests for `backend.rag.retrievers.hybrid_retriever`."""

from __future__ import annotations

import pytest

from backend.core.exceptions import IndexNotFoundError
from backend.rag.retrievers.bm25_retriever import BM25RetrieverWrapper
from backend.rag.retrievers.hybrid_retriever import HybridRetriever
from backend.rag.retrievers.vector_retriever import VectorRetriever


class TestHybridRetriever:
    def test_raises_when_both_indexes_empty(self, settings, faiss_store, bm25_index) -> None:
        vector = VectorRetriever(faiss_store=faiss_store, settings=settings)
        bm25 = BM25RetrieverWrapper(bm25_index=bm25_index, settings=settings)
        hybrid = HybridRetriever(vector_retriever=vector, bm25_retriever=bm25, settings=settings)

        with pytest.raises(IndexNotFoundError):
            hybrid.retrieve("anything")

    def test_fuses_results_from_both_sources(
        self, settings, faiss_store, bm25_index, sample_documents
    ) -> None:
        faiss_store.add_documents(sample_documents)
        bm25_index.add_documents(sample_documents)

        vector = VectorRetriever(faiss_store=faiss_store, settings=settings)
        bm25 = BM25RetrieverWrapper(bm25_index=bm25_index, settings=settings)
        hybrid = HybridRetriever(vector_retriever=vector, bm25_retriever=bm25, settings=settings)

        results = hybrid.retrieve("refund policy return", k=3)

        assert len(results) > 0
        assert all(hasattr(r, "score") and hasattr(r, "document") for r in results)
        # Results should be sorted descending by fused score.
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_scores_are_native_python_floats(
        self, settings, faiss_store, bm25_index, sample_documents
    ) -> None:
        """Regression test: FAISS/BM25 can return numpy scalar types, which
        are not JSON-serializable. ScoredDocument must coerce to float."""
        faiss_store.add_documents(sample_documents)
        bm25_index.add_documents(sample_documents)
        hybrid = HybridRetriever(
            vector_retriever=VectorRetriever(faiss_store=faiss_store, settings=settings),
            bm25_retriever=BM25RetrieverWrapper(bm25_index=bm25_index, settings=settings),
            settings=settings,
        )

        results = hybrid.retrieve("refund policy", k=3)

        for result in results:
            assert type(result.score) is float
            import json

            json.dumps(result.score)  # must not raise

    def test_degrades_gracefully_when_bm25_empty(
        self, settings, faiss_store, bm25_index, sample_documents
    ) -> None:
        """Only the vector index has documents; BM25 legitimately empty."""
        faiss_store.add_documents(sample_documents)
        hybrid = HybridRetriever(
            vector_retriever=VectorRetriever(faiss_store=faiss_store, settings=settings),
            bm25_retriever=BM25RetrieverWrapper(bm25_index=bm25_index, settings=settings),
            settings=settings,
        )

        results = hybrid.retrieve("refund policy", k=3)

        assert len(results) > 0

    def test_respects_k_limit(self, settings, faiss_store, bm25_index, sample_documents) -> None:
        faiss_store.add_documents(sample_documents)
        bm25_index.add_documents(sample_documents)
        hybrid = HybridRetriever(
            vector_retriever=VectorRetriever(faiss_store=faiss_store, settings=settings),
            bm25_retriever=BM25RetrieverWrapper(bm25_index=bm25_index, settings=settings),
            settings=settings,
        )

        results = hybrid.retrieve("policy", k=1)

        assert len(results) == 1
