"""Router node.

The first stop for every question. Decides which branch of the graph to
take:
    * `vectorstore`  — the default; retrieve from the local FAISS/BM25
      indexes.
    * `web_search`   — no documents are indexed yet, or the question looks
      like it needs current/public information.
    * `direct_answer` — greetings, small talk, or meta-questions that need
      no retrieval at all.

If no documents have been indexed at all, the LLM classification call is
skipped entirely (there is nothing for `vectorstore` to plausibly mean),
saving a round-trip and avoiding a confusing "relevant" grade against an
empty index.
"""

from __future__ import annotations

import logging

from langchain_core.output_parsers import StrOutputParser

from backend.config.constants import GraphRoute
from backend.config.settings import get_settings
from backend.core.graph.state import GraphState
from backend.rag.indexing.faiss_store import get_faiss_store
from backend.rag.llms.groq_provider import get_groq_provider
from backend.rag.prompts.router_prompts import ROUTER_PROMPT

logger = logging.getLogger(__name__)

_VALID_ROUTES = {route.value for route in GraphRoute}


def route_question(state: GraphState) -> dict[str, str]:
    """Classify the question and decide which branch of the graph to take.

    Args:
        state: Current graph state.

    Returns:
        A partial state update setting `route`.
    """
    settings = get_settings()
    faiss_store = get_faiss_store()

    if faiss_store.document_count == 0:
        route = (
            GraphRoute.WEB_SEARCH.value
            if settings.tavily_enabled
            else GraphRoute.DIRECT_ANSWER.value
        )
        logger.info("No documents indexed; routing directly to '%s'.", route)
        return {"route": route}

    try:
        llm = get_groq_provider().get_llm(temperature=0.0)
        chain = ROUTER_PROMPT | llm | StrOutputParser()
        raw_output = chain.invoke({"question": state.original_question}).strip().lower()
    except Exception as exc:  # noqa: BLE001 - default to the safe, default route
        logger.warning("Router LLM call failed (%s); defaulting to 'vectorstore'.", exc)
        raw_output = GraphRoute.VECTORSTORE.value

    route = raw_output if raw_output in _VALID_ROUTES else GraphRoute.VECTORSTORE.value
    logger.info("Routed question to '%s'.", route)
    return {"route": route}


def route_after_classification(state: GraphState) -> str:
    """Conditional-edge function: map `state.route` to the next node name.

    Args:
        state: Current graph state (already updated by `route_question`).

    Returns:
        The route value itself, used as the key into the conditional edge
        mapping defined in `backend.core.graph.builder`.
    """
    return state.route
