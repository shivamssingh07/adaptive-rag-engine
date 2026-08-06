"""Adaptive retrieval strategy selection.

Rather than always running the same fixed retrieval chain, `AdaptiveRetriever`
inspects each query's surface characteristics and routes it to the
retrieval strategy best suited to it. This is retrieval-level adaptivity;
it is distinct from (and used underneath) the graph-level adaptive
routing/self-correction loop built in Phase 6.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

from backend.config.settings import Settings, get_settings
from backend.rag.retrievers.base import ScoredDocument
from backend.rag.retrievers.hybrid_retriever import HybridRetriever
from backend.rag.retrievers.mmr_retriever import MMRRetriever
from backend.rag.retrievers.multi_query_retriever import MultiQueryRetriever
from backend.rag.retrievers.self_query_retriever import SelfQueryRetriever

logger = logging.getLogger(__name__)


class _StrategyRetriever(Protocol):
    """Structural interface shared by every retrieval strategy this class
    can dispatch to — just enough for `AdaptiveRetriever.retrieve` to call
    `.retrieve(query, k=...)` on whichever one was selected."""

    def retrieve(self, query: str, k: int | None = None) -> list[ScoredDocument]: ...


# A quoted phrase, or a phrase like "in the report" / "in document X",
# strongly suggests the user is scoping the question to one specific
# uploaded source.
_FILTER_HINT_PATTERN = re.compile(
    r'"[^"]{2,}"|\bin (the )?(file|document|report|pdf|spreadsheet)\b', re.IGNORECASE
)

# Broad, synthesis-style questions benefit from diversity (MMR) more than
# from narrow top-k relevance.
_BROAD_QUESTION_WORDS = frozenset(
    {"compare", "overview", "summarize", "summary", "difference", "differences", "list", "explain"}
)

# Below this word count, a query is likely under-specified enough that
# paraphrase expansion (multi-query) meaningfully improves recall.
_SHORT_QUERY_WORD_THRESHOLD = 6


class AdaptiveRetriever:
    """Chooses a retrieval strategy per-query based on its surface
    characteristics, rather than always running the same fixed chain.

    Selection heuristics (checked in order, first match wins):
        1. Query references a specific source document / quotes an exact
           phrase → `SelfQueryRetriever` (metadata-filtered search).
        2. Query is broad/synthesis-style ("summarize", "compare", ...)
           → `MMRRetriever` (diversity over pure relevance).
        3. Query is short/under-specified (≤ 6 words) → `MultiQueryRetriever`
           (paraphrase expansion improves recall).
        4. Otherwise → `HybridRetriever` (BM25 + vector), the default,
           cost-effective strategy.
    """

    def __init__(
        self,
        hybrid: HybridRetriever | None = None,
        mmr: MMRRetriever | None = None,
        multi_query: MultiQueryRetriever | None = None,
        self_query: SelfQueryRetriever | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Args:
        hybrid: Default retriever. Defaults to a fresh `HybridRetriever`.
        mmr: Diversity retriever. Defaults to a fresh `MMRRetriever`.
        multi_query: Recall-boosting retriever. Defaults to a fresh
            `MultiQueryRetriever`.
        self_query: Filtered-search retriever. Defaults to a fresh
            `SelfQueryRetriever`.
        settings: Application settings. Defaults to the process-wide
            settings singleton.
        """
        self._hybrid = hybrid or HybridRetriever()
        self._mmr = mmr or MMRRetriever()
        self._multi_query = multi_query or MultiQueryRetriever()
        self._self_query = self_query or SelfQueryRetriever()
        self._settings = settings or get_settings()

    def _select_strategy(self, query: str) -> str:
        """Return the name of the strategy selected for this query."""
        if _FILTER_HINT_PATTERN.search(query):
            return "self_query"
        lowered = query.lower()
        if any(word in lowered for word in _BROAD_QUESTION_WORDS):
            return "mmr"
        if len(query.split()) <= _SHORT_QUERY_WORD_THRESHOLD:
            return "multi_query"
        return "hybrid"

    def retrieve(self, query: str, k: int | None = None) -> tuple[list[ScoredDocument], str]:
        """Retrieve documents using the strategy selected for this query.

        Args:
            query: The user's search query.
            k: Number of results to return. Defaults to
                `settings.top_k_retrieval`.

        Returns:
            A tuple of `(results, strategy_name)` — the strategy name is
            surfaced so it can be logged and returned to the client for
            transparency about how the adaptive system behaved for this
            request.

        Raises:
            IndexNotFoundError: If no documents have been indexed yet.
        """
        strategy = self._select_strategy(query)
        logger.info("Adaptive retrieval selected strategy='%s'.", strategy)

        strategy_map: dict[str, _StrategyRetriever] = {
            "self_query": self._self_query,
            "mmr": self._mmr,
            "multi_query": self._multi_query,
            "hybrid": self._hybrid,
        }
        results = strategy_map[strategy].retrieve(query, k=k)
        return results, strategy
