"""Prompt templates for the graph's self-correction grading nodes.

Two independent graders keep the "adaptive" corrective-RAG loop honest:
    * `DOCUMENT_GRADE_PROMPT` — is the retrieved/compressed context actually
      relevant to the question at all? Drives the rewrite-and-retry loop.
    * `GROUNDEDNESS_GRADE_PROMPT` — is the generated answer actually
      supported by that context (not hallucinated)? Drives the
      regenerate-and-retry loop.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

DOCUMENT_GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict grader assessing whether retrieved context is "
            "relevant to a user's question. Respond with EXACTLY one word: "
            "'yes' if the context contains information that helps answer the "
            "question, or 'no' if it does not. If the context is empty or "
            "off-topic, respond 'no'.",
        ),
        (
            "human",
            "Question: {question}\n\nRetrieved context:\n{document}\n\nRelevant (yes/no):",
        ),
    ]
)

GROUNDEDNESS_GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict grader assessing whether a generated answer is "
            "factually grounded in (fully supported by) the provided context. "
            "Respond with EXACTLY one word: 'yes' if every factual claim in "
            "the answer is supported by the context, or 'no' if the answer "
            "contains claims not found in the context.",
        ),
        (
            "human",
            "Context:\n{context}\n\nGenerated answer:\n{answer}\n\nGrounded (yes/no):",
        ),
    ]
)
