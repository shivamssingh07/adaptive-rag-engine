"""RAG subsystem package: LLM providers, embeddings, vector/lexical
indexing, retrievers, rerankers, memory, document loaders, and prompts.

Everything under this package is a reusable building block consumed by the
LangGraph orchestration core (`backend.core.graph`, built in Phase 6). None
of these modules depend on FastAPI or Streamlit, so each is independently
unit-testable.
"""
