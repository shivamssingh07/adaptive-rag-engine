"""Integration tests for the fully compiled adaptive RAG graph
(`backend.core.graph.builder.build_graph`), exercising real routing,
retrieval, grading, and retry-loop behavior end-to-end — not just
individual nodes in isolation (see `tests/unit/test_graph_nodes.py` for
that)."""

from __future__ import annotations

import backend.rag.indexing.bm25_index as bm25_module
import backend.rag.indexing.faiss_store as faiss_module
import backend.rag.llms.groq_provider as groq_module
import backend.rag.rerankers.cross_encoder_reranker as reranker_module
from backend.core.graph.builder import build_graph
from backend.core.graph.state import GraphState


def _wire(faiss_store, bm25_index, reranker, groq_provider) -> None:
    faiss_module._store_singleton = faiss_store
    bm25_module._index_singleton = bm25_index
    reranker_module._reranker_singleton = reranker
    groq_module._provider_singleton = groq_provider


class TestFullGraphRun:
    def test_happy_path_relevant_and_grounded(
        self,
        settings,
        faiss_store,
        bm25_index,
        fake_reranker,
        sample_documents,
        make_fake_groq_provider,
    ) -> None:
        faiss_store.add_documents(sample_documents)
        bm25_index.add_documents(sample_documents)
        _wire(
            faiss_store,
            bm25_index,
            fake_reranker,
            make_fake_groq_provider(
                [
                    "vectorstore",  # router
                    "The refund policy allows returns within 30 days for a full refund.",  # compression 1
                    "International orders are eligible for store credit only.",  # compression 2
                    "NO_RELEVANT_CONTENT",  # compression 3 (cafeteria doc — genuinely irrelevant)
                    "yes",  # grade_documents
                    "You can return items within 30 days for a full refund.",  # generate
                    "yes",  # grade_generation
                ]
            ),
        )
        graph = build_graph()

        result = graph.invoke(
            GraphState(
                session_id="s1",
                original_question="What is the company's official refund policy for returned merchandise?",
            )
        )

        assert result["route"] == "vectorstore"
        assert result["relevance_grade"] == "relevant"
        assert result["groundedness_grade"] == "grounded"
        assert result["retry_count_documents"] == 0
        assert result["retry_count_groundedness"] == 0
        assert "refund" in result["generation"].lower()
        assert len(result["retrieved_documents"]) > 0

    def test_no_documents_routes_direct_answer(
        self, settings, faiss_store, bm25_index, fake_reranker, make_fake_groq_provider
    ) -> None:
        _wire(
            faiss_store,
            bm25_index,
            fake_reranker,
            make_fake_groq_provider(["Hello! I don't have any documents loaded yet."]),
        )
        graph = build_graph()

        result = graph.invoke(GraphState(session_id="s2", original_question="Hi there!"))

        assert result["route"] == "direct_answer"
        assert result["groundedness_grade"] == "grounded"  # skipped, no context
        assert result["generation"]

    def test_irrelevant_documents_exhausts_retries_then_generates(
        self,
        settings,
        faiss_store,
        bm25_index,
        fake_reranker,
        sample_documents,
        make_fake_groq_provider,
    ) -> None:
        faiss_store.add_documents(sample_documents)
        bm25_index.add_documents(sample_documents)
        max_retries = settings.max_document_grade_retries

        responses = ["vectorstore"]
        for attempt in range(max_retries + 1):
            responses += ["irrelevant snippet A", "irrelevant snippet B", "no"]
            if attempt < max_retries:
                # Must be long enough (> 6 words) to avoid the adaptive
                # retriever's multi_query strategy triggering an extra
                # LLM call on the next iteration.
                responses += ["a much longer and more specific rewritten search query here"]
        responses += ["Best effort answer with low confidence.", "yes"]

        _wire(faiss_store, bm25_index, fake_reranker, make_fake_groq_provider(responses))
        graph = build_graph()

        result = graph.invoke(
            GraphState(
                session_id="s3",
                original_question="What is the current weather forecast on the planet Mars today?",
            )
        )

        assert result["retry_count_documents"] == max_retries
        assert result["relevance_grade"] == "irrelevant"
        assert result["generation"]  # still produces a best-effort answer, never crashes

    def test_token_usage_defaults_to_zero_with_fake_llm(
        self, settings, faiss_store, bm25_index, fake_reranker, make_fake_groq_provider
    ) -> None:
        """FakeListChatModel doesn't set usage_metadata; the generate node
        must default gracefully rather than raising an AttributeError."""
        _wire(faiss_store, bm25_index, fake_reranker, make_fake_groq_provider(["A direct answer."]))
        graph = build_graph()

        result = graph.invoke(GraphState(session_id="s4", original_question="Hello!"))

        assert result["token_usage"] == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
