"""Contextual compression: LLM-extracted relevant excerpts.

Retrieved chunks are frequently only partially relevant to the query —
one sentence answers it, the rest is surrounding context. This module asks
the LLM to extract just the relevant sentences from each chunk (verbatim,
not paraphrased), dropping chunks with no relevant content entirely. This
reduces prompt size and noise passed to the final generation step.

Note: this makes one LLM call per input document, sequentially. This is a
deliberate simplicity/latency trade-off appropriate for the small (top-k)
candidate sets this is applied to; a documented future improvement is to
batch these calls concurrently (see `docs/roadmap.md`, added in Phase 10).
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from backend.rag.llms.groq_provider import GroqLLMProvider, get_groq_provider
from backend.rag.prompts.rewrite_prompts import COMPRESSION_PROMPT

logger = logging.getLogger(__name__)

_NO_RELEVANT_CONTENT_MARKER = "NO_RELEVANT_CONTENT"


class ContextualCompressor:
    """Extracts only the query-relevant portion of each retrieved chunk via
    the LLM, dropping chunks with no relevant content."""

    def __init__(self, llm_provider: GroqLLMProvider | None = None) -> None:
        """Args:
        llm_provider: LLM provider used for extraction. Defaults to the
            process-wide Groq provider.
        """
        self._llm_provider = llm_provider or get_groq_provider()

    def compress(self, query: str, documents: list[Document]) -> list[Document]:
        """Compress each document to only its query-relevant content.

        On a per-document LLM failure, the original (uncompressed) chunk
        is kept rather than dropped, so a single flaky call doesn't starve
        the generation step of context.

        Args:
            query: The user's search query.
            documents: Candidate documents to compress (typically the
                reranked top-k).

        Returns:
            A list of `Document` objects — usually shorter than the input,
            since chunks judged entirely irrelevant are dropped and the
            rest are trimmed to their relevant sentences. Original
            metadata is preserved on every kept document.
        """
        if not documents:
            return []

        try:
            llm = self._llm_provider.get_llm(temperature=0.0)
            chain = COMPRESSION_PROMPT | llm | StrOutputParser()
        except Exception as exc:  # noqa: BLE001 - degrade to uncompressed rather than crash the caller
            logger.warning(
                "Could not initialize the compression LLM (%s); returning documents uncompressed.",
                exc,
            )
            return list(documents)

        compressed: list[Document] = []
        for document in documents:
            try:
                extracted = chain.invoke(
                    {"question": query, "context": document.page_content}
                ).strip()
            except Exception as exc:  # noqa: BLE001 - keep original chunk on failure
                logger.warning(
                    "Contextual compression failed for a chunk (%s); keeping it uncompressed.",
                    exc,
                )
                compressed.append(document)
                continue

            if not extracted or _NO_RELEVANT_CONTENT_MARKER in extracted:
                continue

            compressed.append(Document(page_content=extracted, metadata=document.metadata))

        logger.debug(
            "Contextual compression: %d input chunk(s) -> %d retained.",
            len(documents),
            len(compressed),
        )
        return compressed
