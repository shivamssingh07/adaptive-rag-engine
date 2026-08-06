"""Prompt templates for the LLM-driven retrieval components.

Kept in one module (rather than inline in each retriever) so prompt
engineering iteration doesn't require touching the retrieval logic, and so
these prompts can be reused by the graph nodes built in Phase 6.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query rewriting assistant for a retrieval-augmented "
            "generation system. Rewrite the user's question into a single, "
            "clearer, more specific standalone search query that will retrieve "
            "better results from a document search index. Resolve any pronouns "
            "or vague references (e.g. 'it', 'that', 'the previous one') using "
            "the conversation history if provided. Return ONLY the rewritten "
            "query text, with no preamble, quotes, or explanation.",
        ),
        (
            "human",
            "Conversation history:\n{history}\n\nOriginal question: {question}\n\nRewritten query:",
        ),
    ]
)

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You generate alternate phrasings of a search query to improve "
            "document retrieval recall. Given a question, produce {num_queries} "
            "different, diverse phrasings that preserve the original meaning but "
            "vary vocabulary, specificity, and sentence structure. Return ONLY "
            "the phrasings, one per line, with no numbering, bullets, or "
            "explanation.",
        ),
        ("human", "Question: {question}"),
    ]
)

SELF_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract structured metadata filters from a natural-language "
            "search question for a document retrieval system. The only "
            "filterable metadata field is `source`, the exact filename of a "
            'previously uploaded document (e.g. "report.pdf", "notes.docx"). '
            "If the question explicitly names a specific source file, or "
            "clearly attributes a quoted phrase to one, return a JSON object "
            'like {{"source": "report.pdf"}}. If no specific source is named, '
            "return an empty JSON object: {{}}. Return ONLY the JSON object, "
            "with no explanation, code fences, or extra text.",
        ),
        ("human", "Question: {question}"),
    ]
)

COMPRESSION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract only the sentences from a document excerpt that are "
            "directly relevant to answering a given question. Copy the relevant "
            "sentences verbatim — do not paraphrase, summarize, or add "
            "information not present in the excerpt. If NONE of the excerpt is "
            "relevant to the question, respond with exactly: NO_RELEVANT_CONTENT",
        ),
        (
            "human",
            "Question: {question}\n\nDocument excerpt:\n{context}\n\nRelevant sentences:",
        ),
    ]
)
