"""Grade documents node.

The first half of the corrective-RAG self-correction loop: an LLM call
grades whether the compressed context is actually relevant to the
question. If not, the graph either rewrites the query and retries
retrieval (bounded by `settings.max_document_grade_retries`) or, once
retries are exhausted, falls back to web search (if configured) — never
looping forever.
"""

from __future__ import annotations

import logging

from langchain_core.output_parsers import StrOutputParser

from backend.config.constants import RelevanceGrade
from backend.config.settings import get_settings
from backend.core.graph.state import GraphState
from backend.rag.llms.groq_provider import get_groq_provider
from backend.rag.prompts.grading_prompts import DOCUMENT_GRADE_PROMPT

logger = logging.getLogger(__name__)


def grade_documents(state: GraphState) -> dict[str, str]:
    """Grade whether the compressed context is relevant to the question.

    Args:
        state: Current graph state.

    Returns:
        A partial state update setting `relevance_grade`.
    """
    if not state.compressed_documents:
        logger.info("No compressed context to grade; marking as irrelevant.")
        return {"relevance_grade": RelevanceGrade.IRRELEVANT.value}

    context_text = "\n\n".join(doc.page_content for doc in state.compressed_documents)

    try:
        llm = get_groq_provider().get_llm(temperature=0.0)
        chain = DOCUMENT_GRADE_PROMPT | llm | StrOutputParser()
        raw_output = (
            chain.invoke({"question": state.original_question, "document": context_text})
            .strip()
            .lower()
        )
    except Exception as exc:  # noqa: BLE001 - default to relevant rather than looping forever
        logger.warning("Document relevance grading failed (%s); assuming relevant.", exc)
        raw_output = "yes"

    grade = (
        RelevanceGrade.RELEVANT.value
        if raw_output.startswith("yes")
        else RelevanceGrade.IRRELEVANT.value
    )
    logger.info("Document relevance grade: %s", grade)
    return {"relevance_grade": grade}


def route_after_document_grade(state: GraphState) -> str:
    """Conditional-edge function deciding what happens after grading.

    Args:
        state: Current graph state (already updated by `grade_documents`).

    Returns:
        One of `"generate"`, `"rewrite"`, or `"web_search"`.
    """
    settings = get_settings()

    if state.relevance_grade == RelevanceGrade.RELEVANT.value:
        return "generate"

    if state.retry_count_documents < settings.max_document_grade_retries:
        return "rewrite"

    if settings.tavily_enabled:
        logger.info("Document relevance retries exhausted; falling back to web search.")
        return "web_search"

    logger.info(
        "Document relevance retries exhausted and web search is not configured; "
        "generating a best-effort answer with low-confidence context."
    )
    return "generate"
