"""Prompt template for the graph's router node."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You route a user question to the correct data source for a "
            "retrieval-augmented generation system with access to a private "
            "document collection. Respond with EXACTLY one word:\n"
            "- 'vectorstore' if the question could plausibly be answered from "
            "the uploaded private documents. This is the default for most "
            "document-style or domain-specific questions.\n"
            "- 'web_search' if the question is clearly about current events, "
            "real-time facts, or general public knowledge unlikely to be in "
            "private documents.\n"
            "- 'direct_answer' if the question is a greeting, small talk, or a "
            "meta-question about the assistant itself that needs no document "
            "lookup at all.\n"
            "Respond with ONLY one of: vectorstore, web_search, direct_answer",
        ),
        ("human", "Question: {question}"),
    ]
)
