"""Grade generation node.

The second half of the self-correction loop: checks whether the generated
answer is actually supported by the context it was given (i.e. not
hallucinated). Grading is skipped — and the answer is trusted — when there
was no context to begin with (the `direct_answer` route, or every upstream
retrieval/web-search attempt came up empty): there is nothing to check
groundedness against, and demanding it would only produce an infinite
regeneration loop for a question the system correctly has no source
material for.
"""

from __future__ import annotations

import logging

from langchain_core.output_parsers import StrOutputParser

from backend.config.constants import GroundednessGrade
from backend.config.settings import get_settings
from backend.core.graph.state import GraphState
from backend.rag.llms.groq_provider import get_groq_provider
from backend.rag.prompts.grading_prompts import GROUNDEDNESS_GRADE_PROMPT

logger = logging.getLogger(__name__)


def grade_generation(state: GraphState) -> dict[str, object]:
    """Grade whether the generated answer is grounded in its context.

    Args:
        state: Current graph state.

    Returns:
        A partial state update setting `groundedness_grade`, and, if the
        answer is not grounded, incrementing `retry_count_groundedness`.
    """
    if not state.compressed_documents:
        logger.debug("No context was used for generation; skipping groundedness check.")
        return {"groundedness_grade": GroundednessGrade.GROUNDED.value}

    context_text = "\n\n".join(doc.page_content for doc in state.compressed_documents)

    try:
        llm = get_groq_provider().get_llm(temperature=0.0)
        chain = GROUNDEDNESS_GRADE_PROMPT | llm | StrOutputParser()
        raw_output = (
            chain.invoke({"context": context_text, "answer": state.generation}).strip().lower()
        )
    except Exception as exc:  # noqa: BLE001 - default to grounded rather than looping forever
        logger.warning("Groundedness grading failed (%s); assuming grounded.", exc)
        raw_output = "yes"

    if raw_output.startswith("yes"):
        logger.info("Groundedness grade: grounded.")
        return {"groundedness_grade": GroundednessGrade.GROUNDED.value}

    attempt = state.retry_count_groundedness + 1
    logger.info("Groundedness grade: NOT grounded (retry %d).", attempt)
    return {
        "groundedness_grade": GroundednessGrade.NOT_GROUNDED.value,
        "retry_count_groundedness": attempt,
    }


def route_after_groundedness(state: GraphState) -> str:
    """Conditional-edge function deciding what happens after grading.

    Args:
        state: Current graph state (already updated by `grade_generation`).

    Returns:
        `"retry"` to regenerate, or `"end"` to return the current answer
        (either because it's grounded, or because retries are exhausted
        and a best-effort answer is returned rather than looping forever).
    """
    settings = get_settings()

    if state.groundedness_grade == GroundednessGrade.GROUNDED.value:
        return "end"

    if state.retry_count_groundedness <= settings.max_groundedness_retries:
        return "retry"

    logger.info("Groundedness retries exhausted; returning best-effort answer.")
    return "end"
