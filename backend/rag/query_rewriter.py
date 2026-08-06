"""Standalone query rewriting.

Distinct from `MultiQueryRetriever` (which generates several parallel
paraphrases): `QueryRewriter` produces a single, clearer, standalone
version of the query — resolving conversational references ("what about
its pricing?" → "what is Product X's pricing?") using recent chat history.
Used both directly and as the graph's `rewrite_query` node in Phase 6.
"""

from __future__ import annotations

import logging

from langchain_core.output_parsers import StrOutputParser

from backend.rag.llms.groq_provider import GroqLLMProvider, get_groq_provider
from backend.rag.prompts.rewrite_prompts import QUERY_REWRITE_PROMPT

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Rewrites a (possibly ambiguous or conversational) user question into
    a clearer, standalone search query."""

    def __init__(self, llm_provider: GroqLLMProvider | None = None) -> None:
        """Args:
        llm_provider: LLM provider used for rewriting. Defaults to the
            process-wide Groq provider.
        """
        self._llm_provider = llm_provider or get_groq_provider()

    def rewrite(self, question: str, history: str = "") -> str:
        """Rewrite a question into a standalone search query.

        Falls back to returning the original question, unmodified, if the
        LLM call fails — rewriting is an optimization, not a hard
        dependency for retrieval to function.

        Args:
            question: The user's original question, as typed.
            history: A plain-text summary of recent conversation turns,
                used to resolve pronouns/references. Pass an empty string
                for the first turn of a conversation.

        Returns:
            The rewritten, standalone query, or the original question if
            rewriting failed or produced empty output.
        """
        try:
            llm = self._llm_provider.get_llm(temperature=0.0)
            chain = QUERY_REWRITE_PROMPT | llm | StrOutputParser()
            rewritten = chain.invoke(
                {"question": question, "history": history or "(no prior conversation)"}
            )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.warning(
                "Query rewriting failed (%s); falling back to the original question.", exc
            )
            return question

        rewritten = rewritten.strip().strip('"')
        return rewritten or question
