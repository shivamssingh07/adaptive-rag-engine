"""Web search node.

Invoked either directly (router decided the question needs public/current
information, or no documents are indexed at all) or as a fallback after
local retrieval retries are exhausted. Fully optional: if Tavily is not
configured, this node degrades to a no-op (state passes through
unmodified, `used_web_search` stays `False`) rather than raising —
`generate` then proceeds with whatever local context (if any) is already
in state.
"""

from __future__ import annotations

import logging

from backend.config.settings import get_settings
from backend.core.graph.state import GraphState
from backend.rag.retrievers.base import ScoredDocument
from backend.rag.search.tavily_search import get_tavily_search

logger = logging.getLogger(__name__)


def web_search(state: GraphState) -> dict[str, object]:
    """Fetch web search results and merge them into the retrieval context.

    Args:
        state: Current graph state.

    Returns:
        A partial state update. If Tavily is unavailable or returns no
        usable results, only `used_web_search: False` is set and existing
        `retrieved_documents`/`compressed_documents` are left untouched.
        Otherwise, web results are appended to both.
    """
    settings = get_settings()
    if not settings.tavily_enabled:
        logger.info("Web search node reached but Tavily is not configured; skipping.")
        return {"used_web_search": False}

    tavily = get_tavily_search()
    web_documents = tavily.search(state.original_question)

    if not web_documents:
        logger.info("Web search returned no usable results.")
        return {"used_web_search": False}

    web_scored = [ScoredDocument(document=doc, score=1.0) for doc in web_documents]
    logger.info("Web search returned %d result(s); merged into context.", len(web_documents))
    return {
        "retrieved_documents": [*state.retrieved_documents, *web_scored],
        "compressed_documents": [*state.compressed_documents, *web_documents],
        "used_web_search": True,
    }
