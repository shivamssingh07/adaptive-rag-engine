"""Adaptive RAG graph construction.

Wires together every node in `backend.core.graph.nodes` into the compiled
LangGraph state machine described in the Phase 1 architecture doc:

    START -> route_question -> {retrieve | web_search | generate}
    retrieve -> grade_documents -> {generate | rewrite_query | web_search}
    rewrite_query -> retrieve
    web_search -> generate
    generate -> grade_generation -> {END | generate (retry)}

The compiled graph is cached as a process-wide singleton (graph
compilation has non-trivial overhead and the graph structure never
changes at runtime).
"""

from __future__ import annotations

import threading
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from backend.config.constants import GraphNode
from backend.core.graph.nodes import (
    generate,
    grade_documents,
    grade_generation,
    retrieve,
    rewrite_query,
    router,
    web_search,
)
from backend.core.graph.state import GraphState


def build_graph() -> CompiledStateGraph[GraphState, Any, Any, Any]:
    """Construct and compile the adaptive RAG state graph.

    Returns:
        A freshly-compiled `CompiledStateGraph`. Prefer
        `get_compiled_graph()` over calling this directly, to reuse the
        process-wide singleton.
    """
    graph = StateGraph(GraphState)

    graph.add_node(GraphNode.ROUTE_QUESTION.value, router.route_question)
    graph.add_node(GraphNode.RETRIEVE.value, retrieve.retrieve)
    graph.add_node(GraphNode.GRADE_DOCUMENTS.value, grade_documents.grade_documents)
    graph.add_node(GraphNode.REWRITE_QUERY.value, rewrite_query.rewrite_query)
    graph.add_node(GraphNode.WEB_SEARCH.value, web_search.web_search)
    graph.add_node(GraphNode.GENERATE.value, generate.generate)
    graph.add_node(GraphNode.GRADE_GENERATION.value, grade_generation.grade_generation)

    graph.add_edge(START, GraphNode.ROUTE_QUESTION.value)

    graph.add_conditional_edges(
        GraphNode.ROUTE_QUESTION.value,
        router.route_after_classification,
        {
            "vectorstore": GraphNode.RETRIEVE.value,
            "web_search": GraphNode.WEB_SEARCH.value,
            "direct_answer": GraphNode.GENERATE.value,
        },
    )

    graph.add_edge(GraphNode.RETRIEVE.value, GraphNode.GRADE_DOCUMENTS.value)

    graph.add_conditional_edges(
        GraphNode.GRADE_DOCUMENTS.value,
        grade_documents.route_after_document_grade,
        {
            "generate": GraphNode.GENERATE.value,
            "rewrite": GraphNode.REWRITE_QUERY.value,
            "web_search": GraphNode.WEB_SEARCH.value,
        },
    )

    graph.add_edge(GraphNode.REWRITE_QUERY.value, GraphNode.RETRIEVE.value)
    graph.add_edge(GraphNode.WEB_SEARCH.value, GraphNode.GENERATE.value)
    graph.add_edge(GraphNode.GENERATE.value, GraphNode.GRADE_GENERATION.value)

    graph.add_conditional_edges(
        GraphNode.GRADE_GENERATION.value,
        grade_generation.route_after_groundedness,
        {
            "end": END,
            "retry": GraphNode.GENERATE.value,
        },
    )

    return graph.compile()


_compiled_graph: CompiledStateGraph[GraphState, Any, Any, Any] | None = None
_compiled_graph_lock = threading.Lock()


def get_compiled_graph() -> CompiledStateGraph[GraphState, Any, Any, Any]:
    """Return the process-wide compiled graph singleton.

    Returns:
        The shared `CompiledStateGraph` instance.
    """
    global _compiled_graph
    if _compiled_graph is None:
        with _compiled_graph_lock:
            if _compiled_graph is None:
                _compiled_graph = build_graph()
    return _compiled_graph
