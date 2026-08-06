"""Self-query retrieval: LLM-extracted metadata filtering.

Asks the LLM to identify, from the natural-language query alone, whether
the user is asking about a specific uploaded source document (e.g. "what
does report.pdf say about revenue?"). If so, that filename is applied as a
metadata filter directly against the FAISS store alongside the semantic
search, scoping results to that document. Falls back to unfiltered search
if no filter is detected, the filter matches nothing, or filtering fails.
"""

from __future__ import annotations

import json
import logging

from langchain_core.output_parsers import StrOutputParser

from backend.config.settings import Settings, get_settings
from backend.rag.indexing.faiss_store import FAISSVectorStore, get_faiss_store
from backend.rag.llms.groq_provider import GroqLLMProvider, get_groq_provider
from backend.rag.prompts.rewrite_prompts import SELF_QUERY_PROMPT
from backend.rag.retrievers.base import ScoredDocument

logger = logging.getLogger(__name__)


class SelfQueryRetriever:
    """Extracts a `source` filename filter from the query via the LLM and
    applies it as a metadata filter against the FAISS store."""

    def __init__(
        self,
        faiss_store: FAISSVectorStore | None = None,
        llm_provider: GroqLLMProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Args:
        faiss_store: Vector store to search. Defaults to the process-wide
            singleton.
        llm_provider: LLM provider used to extract the filter. Defaults to
            the process-wide Groq provider.
        settings: Application settings. Defaults to the process-wide
            settings singleton.
        """
        self._store = faiss_store or get_faiss_store()
        self._llm_provider = llm_provider or get_groq_provider()
        self._settings = settings or get_settings()

    def _extract_filter(self, query: str) -> dict[str, str]:
        """Ask the LLM to extract a metadata filter; returns an empty dict
        on any parsing or generation failure rather than raising, since a
        missing filter is a valid (and common) outcome, not an error."""
        try:
            llm = self._llm_provider.get_llm(temperature=0.0)
            chain = SELF_QUERY_PROMPT | llm | StrOutputParser()
            raw_output = chain.invoke({"question": query})
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.warning(
                "Self-query filter extraction failed (%s); proceeding without a filter.",
                exc,
            )
            return {}

        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if "\n" in cleaned:
                cleaned = cleaned.split("\n", 1)[1]

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.debug(
                "Self-query filter extraction returned non-JSON output; "
                "proceeding without a filter."
            )
            return {}

        if not isinstance(parsed, dict):
            return {}
        return {str(k): str(v) for k, v in parsed.items() if v not in (None, "", "null")}

    def retrieve(self, query: str, k: int | None = None) -> list[ScoredDocument]:
        """Retrieve chunks, applying an LLM-extracted metadata filter when
        one can be confidently identified.

        Args:
            query: The user's search query.
            k: Number of results to return. Defaults to
                `settings.top_k_retrieval`.

        Returns:
            Scored documents sorted by descending similarity. Filtered to
            a single source document when a filter was detected and
            matched at least one chunk; unfiltered otherwise.

        Raises:
            IndexNotFoundError: If no documents have been indexed yet.
            RetrievalError: If the underlying search fails.
        """
        effective_k = k or self._settings.top_k_retrieval
        metadata_filter = self._extract_filter(query)

        if metadata_filter:
            logger.debug("Self-query extracted metadata filter: %s", metadata_filter)
            try:
                filtered_results = self._store.similarity_search_with_score(
                    query, k=effective_k, filter=metadata_filter
                )
            except Exception as exc:  # noqa: BLE001 - fall back to unfiltered search
                logger.warning(
                    "Self-query filtered search failed (%s); falling back to unfiltered search.",
                    exc,
                )
                filtered_results = []

            if filtered_results:
                return [
                    ScoredDocument(document=doc, score=1.0 / (1.0 + distance))
                    for doc, distance in filtered_results
                ]
            logger.debug(
                "Self-query filter matched no documents; falling back to unfiltered search."
            )

        unfiltered_results = self._store.similarity_search_with_score(query, k=effective_k)
        return [
            ScoredDocument(document=doc, score=1.0 / (1.0 + distance))
            for doc, distance in unfiltered_results
        ]
