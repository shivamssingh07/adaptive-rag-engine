"""Multi-query retrieval: LLM-generated query paraphrase expansion.

A single phrasing of a question can miss chunks that use different
vocabulary for the same concept. This retriever asks the LLM to generate
several diverse paraphrases of the query, runs hybrid retrieval for each,
and merges the results — keeping, for any chunk retrieved by more than one
variant, its highest score.
"""

from __future__ import annotations

import logging

from langchain_core.output_parsers import StrOutputParser

from backend.config.settings import Settings, get_settings
from backend.rag.llms.groq_provider import GroqLLMProvider, get_groq_provider
from backend.rag.prompts.rewrite_prompts import MULTI_QUERY_PROMPT
from backend.rag.retrievers.base import ScoredDocument, document_key
from backend.rag.retrievers.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


class MultiQueryRetriever:
    """Expands a query into multiple LLM-generated paraphrases and merges
    hybrid retrieval results across all of them."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        llm_provider: GroqLLMProvider | None = None,
        settings: Settings | None = None,
        num_queries: int = 3,
    ) -> None:
        """Args:
        retriever: Underlying retriever run for each query variant.
            Defaults to a fresh `HybridRetriever`.
        llm_provider: LLM provider used to generate paraphrases. Defaults
            to the process-wide Groq provider.
        settings: Application settings. Defaults to the process-wide
            settings singleton.
        num_queries: Number of paraphrases to generate in addition to the
            original query.
        """
        self._retriever = retriever or HybridRetriever()
        self._llm_provider = llm_provider or get_groq_provider()
        self._settings = settings or get_settings()
        self._num_queries = num_queries

    def _generate_query_variants(self, query: str) -> list[str]:
        """Ask the LLM for paraphrased variants; degrade to just the
        original query on any failure rather than raising."""
        try:
            llm = self._llm_provider.get_llm(temperature=0.3)
            chain = MULTI_QUERY_PROMPT | llm | StrOutputParser()
            raw_output = chain.invoke({"question": query, "num_queries": self._num_queries})
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.warning(
                "Multi-query variant generation failed (%s); using the original query only.",
                exc,
            )
            return []

        variants = [
            line.strip("-•*0123456789. \t")
            for line in raw_output.strip().splitlines()
            if line.strip()
        ]
        return [v for v in variants if v][: self._num_queries]

    def retrieve(self, query: str, k: int | None = None) -> list[ScoredDocument]:
        """Retrieve using the original query plus LLM-generated variants,
        merging and de-duplicating results by chunk.

        Args:
            query: The user's original search query.
            k: Number of final results to return. Defaults to
                `settings.top_k_retrieval`.

        Returns:
            Scored documents sorted by descending best-observed score
            across all query variants.

        Raises:
            IndexNotFoundError: If no documents have been indexed yet
                (raised only if every query variant fails to retrieve).
        """
        effective_k = k or self._settings.top_k_retrieval
        variants = self._generate_query_variants(query)
        all_queries = [query, *variants]
        logger.debug("Multi-query variants for retrieval: %s", all_queries)

        merged: dict[str, ScoredDocument] = {}
        last_error: Exception | None = None

        for variant in all_queries:
            try:
                results = self._retriever.retrieve(variant, k=effective_k)
            except Exception as exc:  # noqa: BLE001 - try remaining variants
                last_error = exc
                logger.debug("Retrieval failed for query variant '%s': %s", variant, exc)
                continue
            for scored in results:
                key = document_key(scored.document)
                if key not in merged or scored.score > merged[key].score:
                    merged[key] = scored

        if not merged and last_error is not None:
            raise last_error

        ranked = sorted(merged.values(), key=lambda s: s.score, reverse=True)
        return ranked[:effective_k]
