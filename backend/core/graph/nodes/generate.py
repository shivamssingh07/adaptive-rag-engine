"""Generate node.

Produces the final answer from whatever context is in state at this point
— compressed local document chunks, web search results, both, or neither
(for the `direct_answer` route, or when everything upstream came up
empty). The prompt explicitly instructs the model to say so honestly
rather than fabricate an answer when context is insufficient.

This node runs to full completion (not token-streamed) because its output
may still need to pass the groundedness grader and potentially be
regenerated — see `backend.core.graph.nodes.grade_generation`. The API
layer streams the graph's *final*, verified answer to the client in
chunks; see `backend.api.routes.chat` for that transport-level streaming.

Unlike every other LLM call in this graph, this node invokes the raw chat
model directly (not through `StrOutputParser`) so it can read
`usage_metadata` off the returned `AIMessage` for observability — token
counts are surfaced in the final `/chat` API response.
"""

from __future__ import annotations

import logging

from backend.core.exceptions import GenerationError
from backend.core.graph.state import GraphState
from backend.rag.llms.groq_provider import get_groq_provider
from backend.rag.prompts.generation_prompts import GENERATION_PROMPT

logger = logging.getLogger(__name__)

_NO_CONTEXT_PLACEHOLDER = "(no relevant context was found for this question)"


def generate(state: GraphState) -> dict[str, object]:
    """Generate the final answer text.

    Args:
        state: Current graph state.

    Returns:
        A partial state update setting `generation` and `token_usage`.

    Raises:
        GenerationError: If the LLM call fails.
    """
    context_text = (
        "\n\n---\n\n".join(doc.page_content for doc in state.compressed_documents)
        if state.compressed_documents
        else _NO_CONTEXT_PLACEHOLDER
    )

    try:
        llm = get_groq_provider().get_llm()
        chain = GENERATION_PROMPT | llm
        response = chain.invoke(
            {
                "question": state.original_question,
                "context": context_text,
                "history": state.history_text or "(no prior conversation)",
            }
        )
    except Exception as exc:  # noqa: BLE001 - wrapped as a domain error
        raise GenerationError(f"Answer generation failed: {exc}") from exc

    answer = str(response.content).strip()

    # `usage_metadata` is populated by langchain-groq on real API responses
    # but not by test doubles like FakeListChatModel; default to an empty
    # dict rather than letting a missing attribute break generation.
    raw_usage = getattr(response, "usage_metadata", None) or {}
    token_usage = {
        "input_tokens": int(raw_usage.get("input_tokens", 0)),
        "output_tokens": int(raw_usage.get("output_tokens", 0)),
        "total_tokens": int(raw_usage.get("total_tokens", 0)),
    }

    logger.info("Generated answer (%d characters, %s).", len(answer), token_usage)
    return {"generation": answer, "token_usage": token_usage}
