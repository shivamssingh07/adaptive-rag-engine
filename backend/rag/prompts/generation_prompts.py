"""Prompt template for the graph's answer generation node."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful, precise assistant answering questions using the "
            "provided context (drawn from the user's uploaded documents and, "
            "when noted, web search results). Answer accurately and concisely "
            "based on the context. If the context does not contain enough "
            "information to answer confidently, say so honestly rather than "
            "guessing or fabricating information. Use the conversation history "
            "only to understand what the question is referring to — always "
            "answer the current question, not a past one.",
        ),
        (
            "human",
            "Conversation history:\n{history}\n\nContext:\n{context}\n\n"
            "Question: {question}\n\nAnswer:",
        ),
    ]
)
