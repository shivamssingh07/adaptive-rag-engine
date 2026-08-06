"""Retrieve node.

Runs the full first-stage retrieval pipeline in one node:
    1. `AdaptiveRetriever` picks a strategy (hybrid/MMR/multi-query/
       self-query) per-query and returns candidates.
    2. `CrossEncoderReranker` re-scores and truncates to the top-k most
       precisely relevant candidates.
    3. `ContextualCompressor` extracts just the query-relevant sentences
       from each surviving chunk (dropping chunks with none).

Both the reranked-but-uncompressed results (`retrieved_documents`, used
later for citations with clean excerpts) and the compressed text
(`compressed_documents`, used for grading and generation) are kept in
state, associated by `chunk_id`.
"""

from __future__ import annotations

import logging
import math

from backend.config.settings import get_settings
from backend.core.exceptions import IndexNotFoundError
from backend.core.graph.state import GraphState
from backend.rag.rerankers.cross_encoder_reranker import get_cross_encoder_reranker
from backend.rag.retrievers.adaptive_retriever import AdaptiveRetriever
from backend.rag.retrievers.base import ScoredDocument
from backend.rag.retrievers.contextual_compression import ContextualCompressor

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    """Map an unbounded cross-encoder logit into a `(0, 1)` display score."""
    return 1.0 / (1.0 + math.exp(-x))


def retrieve(state: GraphState) -> dict[str, object]:
    """Retrieve, rerank, and compress context for the current query.

    Args:
        state: Current graph state.

    Returns:
        A partial state update setting `retrieved_documents`,
        `compressed_documents`, and `retrieval_strategy`.
    """
    settings = get_settings()
    query = state.effective_query

    adaptive_retriever = AdaptiveRetriever()
    fetch_k = max(settings.top_k_retrieval, settings.top_k_rerank * 3)
    try:
        candidates, strategy = adaptive_retriever.retrieve(query, k=fetch_k)
    except IndexNotFoundError:
        logger.info("No documents indexed; retrieve node returning empty context.")
        return {"retrieved_documents": [], "compressed_documents": [], "retrieval_strategy": "none"}

    if not candidates:
        logger.info("Adaptive retrieval (strategy='%s') returned no candidates.", strategy)
        return {
            "retrieved_documents": [],
            "compressed_documents": [],
            "retrieval_strategy": strategy,
        }

    reranker = get_cross_encoder_reranker()
    reranked_pairs = reranker.rerank(
        query, [candidate.document for candidate in candidates], top_k=settings.top_k_rerank
    )
    reranked_scored = [
        ScoredDocument(document=doc, score=_sigmoid(raw_score)) for doc, raw_score in reranked_pairs
    ]

    compressor = ContextualCompressor()
    compressed = compressor.compress(query, [scored.document for scored in reranked_scored])

    logger.info(
        "Retrieve node: strategy='%s', %d candidate(s) -> %d reranked -> %d after compression.",
        strategy,
        len(candidates),
        len(reranked_scored),
        len(compressed),
    )
    return {
        "retrieved_documents": reranked_scored,
        "compressed_documents": compressed,
        "retrieval_strategy": strategy,
    }
