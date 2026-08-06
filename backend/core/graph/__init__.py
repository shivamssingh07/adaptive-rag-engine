"""LangGraph adaptive RAG orchestration package.

`backend.core.graph.builder.get_compiled_graph()` returns the compiled
state machine described in the Phase 1 architecture doc: route -> retrieve
-> grade -> (rewrite/retry | web search fallback) -> generate -> grade
generation -> (retry | return). Every node is a thin wrapper around the
RAG primitives built in Phase 3 — no retrieval, generation, or scoring
logic lives here, only orchestration.
"""
