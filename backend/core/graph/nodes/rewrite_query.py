"""Rewrite query node.

Invoked when `grade_documents` judges the retrieved context irrelevant and
retries remain. Produces a clearer, standalone rewritten query (using
conversation history to resolve references) and increments the retry
counter that bounds this loop — see
`backend.core.graph.nodes.grade_documents.route_after_document_grade`.
"""

from __future__ import annotations

import logging

from backend.core.graph.state import GraphState
from backend.rag.query_rewriter import QueryRewriter

logger = logging.getLogger(__name__)


def rewrite_query(state: GraphState) -> dict[str, object]:
    """Rewrite the query and bump the document-grade retry counter.

    Args:
        state: Current graph state.

    Returns:
        A partial state update setting `rewritten_query` and incrementing
        `retry_count_documents`.
    """
    rewriter = QueryRewriter()
    new_query = rewriter.rewrite(state.original_question, state.history_text)
    attempt = state.retry_count_documents + 1
    logger.info(
        "Rewriting query (retry %d): '%s' -> '%s'",
        attempt,
        state.original_question,
        new_query,
    )
    return {"rewritten_query": new_query, "retry_count_documents": attempt}
