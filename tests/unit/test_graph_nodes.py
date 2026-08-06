"""Unit tests for individual `backend.core.graph.nodes` functions,
exercised in isolation (not through the full compiled graph — see
`tests/integration/test_full_graph_run.py` for that)."""

from __future__ import annotations

import backend.rag.indexing.bm25_index as bm25_module
import backend.rag.indexing.faiss_store as faiss_module
import backend.rag.llms.groq_provider as groq_module
import backend.rag.rerankers.cross_encoder_reranker as reranker_module
from backend.config.constants import GroundednessGrade, RelevanceGrade
from backend.core.graph.nodes.grade_documents import grade_documents, route_after_document_grade
from backend.core.graph.nodes.grade_generation import grade_generation, route_after_groundedness
from backend.core.graph.nodes.retrieve import retrieve
from backend.core.graph.nodes.rewrite_query import rewrite_query
from backend.core.graph.nodes.router import route_after_classification, route_question
from backend.core.graph.state import GraphState


def _wire_singletons(faiss_store=None, bm25_index=None, groq_provider=None, reranker=None) -> None:
    """Point every module-level singleton the graph nodes reach for at the
    given fakes, restoring isolation between tests."""
    faiss_module._store_singleton = faiss_store
    bm25_module._index_singleton = bm25_index
    groq_module._provider_singleton = groq_provider
    reranker_module._reranker_singleton = reranker


class TestRouterNode:
    def test_routes_to_direct_answer_when_no_documents_and_no_tavily(
        self, settings, faiss_store, bm25_index
    ) -> None:
        _wire_singletons(faiss_store=faiss_store, bm25_index=bm25_index)
        state = GraphState(session_id="s1", original_question="Hello there!")

        result = route_question(state)

        assert result["route"] == "direct_answer"

    def test_routes_via_llm_when_documents_exist(
        self, settings, faiss_store, bm25_index, sample_documents, make_fake_groq_provider
    ) -> None:
        faiss_store.add_documents(sample_documents)
        _wire_singletons(
            faiss_store=faiss_store,
            bm25_index=bm25_index,
            groq_provider=make_fake_groq_provider(["vectorstore"]),
        )
        state = GraphState(session_id="s1", original_question="What is the refund policy?")

        result = route_question(state)

        assert result["route"] == "vectorstore"

    def test_route_after_classification_reads_state(self) -> None:
        state = GraphState(session_id="s1", original_question="q", route="web_search")
        assert route_after_classification(state) == "web_search"


class TestGradeDocumentsNode:
    def test_no_context_is_irrelevant(self, settings) -> None:
        state = GraphState(session_id="s1", original_question="q", compressed_documents=[])
        result = grade_documents(state)
        assert result["relevance_grade"] == RelevanceGrade.IRRELEVANT.value

    def test_relevant_grade_from_llm(
        self, settings, make_fake_groq_provider, sample_documents
    ) -> None:
        _wire_singletons(groq_provider=make_fake_groq_provider(["yes"]))
        state = GraphState(
            session_id="s1",
            original_question="What is the refund policy?",
            compressed_documents=[sample_documents[0]],
        )
        result = grade_documents(state)
        assert result["relevance_grade"] == RelevanceGrade.RELEVANT.value

    def test_route_to_generate_when_relevant(self, settings) -> None:
        state = GraphState(session_id="s1", original_question="q", relevance_grade="relevant")
        assert route_after_document_grade(state) == "generate"

    def test_route_to_rewrite_when_irrelevant_and_retries_remain(self, settings) -> None:
        state = GraphState(
            session_id="s1",
            original_question="q",
            relevance_grade="irrelevant",
            retry_count_documents=0,
        )
        assert route_after_document_grade(state) == "rewrite"

    def test_route_to_generate_when_retries_exhausted_and_no_tavily(self, settings) -> None:
        state = GraphState(
            session_id="s1",
            original_question="q",
            relevance_grade="irrelevant",
            retry_count_documents=settings.max_document_grade_retries,
        )
        assert route_after_document_grade(state) == "generate"


class TestRewriteQueryNode:
    def test_rewrites_and_increments_retry_count(self, settings, make_fake_groq_provider) -> None:
        _wire_singletons(groq_provider=make_fake_groq_provider(["a clearer rewritten question"]))
        state = GraphState(
            session_id="s1", original_question="what about it?", retry_count_documents=0
        )

        result = rewrite_query(state)

        assert result["rewritten_query"] == "a clearer rewritten question"
        assert result["retry_count_documents"] == 1

    def test_falls_back_to_original_on_llm_failure(self, settings) -> None:
        class BrokenProvider:
            def get_llm(self, temperature=None):
                raise RuntimeError("simulated failure")

        _wire_singletons(groq_provider=BrokenProvider())
        state = GraphState(
            session_id="s1", original_question="original question", retry_count_documents=0
        )

        result = rewrite_query(state)

        assert result["rewritten_query"] == "original question"


class TestGradeGenerationNode:
    def test_no_context_skips_grading_and_is_grounded(self, settings) -> None:
        state = GraphState(
            session_id="s1", original_question="q", compressed_documents=[], generation="an answer"
        )
        result = grade_generation(state)
        assert result["groundedness_grade"] == GroundednessGrade.GROUNDED.value

    def test_not_grounded_increments_retry_count(
        self, settings, make_fake_groq_provider, sample_documents
    ) -> None:
        _wire_singletons(groq_provider=make_fake_groq_provider(["no"]))
        state = GraphState(
            session_id="s1",
            original_question="q",
            compressed_documents=[sample_documents[0]],
            generation="a hallucinated answer",
            retry_count_groundedness=0,
        )

        result = grade_generation(state)

        assert result["groundedness_grade"] == GroundednessGrade.NOT_GROUNDED.value
        assert result["retry_count_groundedness"] == 1

    def test_route_to_end_when_grounded(self, settings) -> None:
        state = GraphState(session_id="s1", original_question="q", groundedness_grade="grounded")
        assert route_after_groundedness(state) == "end"

    def test_route_to_retry_when_not_grounded_and_retries_remain(self, settings) -> None:
        state = GraphState(
            session_id="s1",
            original_question="q",
            groundedness_grade="not_grounded",
            retry_count_groundedness=1,
        )
        assert route_after_groundedness(state) == "retry"

    def test_route_to_end_when_retries_exhausted(self, settings) -> None:
        state = GraphState(
            session_id="s1",
            original_question="q",
            groundedness_grade="not_grounded",
            retry_count_groundedness=settings.max_groundedness_retries + 1,
        )
        assert route_after_groundedness(state) == "end"


class TestRetrieveNode:
    def test_empty_index_returns_empty_context(self, settings, faiss_store, bm25_index) -> None:
        _wire_singletons(faiss_store=faiss_store, bm25_index=bm25_index)
        state = GraphState(session_id="s1", original_question="anything at all here")

        result = retrieve(state)

        assert result["retrieved_documents"] == []
        assert result["compressed_documents"] == []
        assert result["retrieval_strategy"] == "none"

    def test_retrieves_reranks_and_compresses(
        self,
        settings,
        faiss_store,
        bm25_index,
        sample_documents,
        fake_reranker,
        make_fake_groq_provider,
    ) -> None:
        faiss_store.add_documents(sample_documents)
        bm25_index.add_documents(sample_documents)
        # One compression LLM call per candidate document survives reranking.
        _wire_singletons(
            faiss_store=faiss_store,
            bm25_index=bm25_index,
            reranker=fake_reranker,
            groq_provider=make_fake_groq_provider(
                ["The refund policy allows returns within 30 days for a full refund."] * 10
            ),
        )
        state = GraphState(
            session_id="s1", original_question="What is the company's refund policy for purchases?"
        )

        result = retrieve(state)

        assert result["retrieval_strategy"] in {"hybrid", "mmr", "multi_query", "self_query"}
        assert len(result["retrieved_documents"]) > 0
