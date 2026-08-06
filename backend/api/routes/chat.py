"""Chat routes.

`POST /chat` runs the full adaptive RAG graph for one turn of a
conversation, then either streams the verified answer via Server-Sent
Events (the default) or returns it as a single JSON payload. See the
module docstring on `backend.core.graph.nodes.generate` for why streaming
happens *after* the graph completes rather than token-by-token through the
self-correction loop.

Also exposes session-scoped conversation management (`GET`/`DELETE`
`/chat/{session_id}` and `GET /chat/{session_id}/export`) to satisfy the
"multiple sessions, conversation export, conversation reset" requirements.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langgraph.graph.state import CompiledStateGraph
from starlette.concurrency import run_in_threadpool

from backend.api.dependencies import get_conversation_memory, get_graph, get_session_store
from backend.api.schemas.chat import (
    ChatMetrics,
    ChatRequest,
    ChatResponse,
    SessionExportResponse,
    SourceCitationSchema,
)
from backend.core.exceptions import SessionNotFoundError
from backend.core.graph.state import GraphState
from backend.rag.citations import build_citations
from backend.rag.memory.conversation_memory import ConversationMemory
from backend.rag.memory.session_store import SessionStore
from backend.utils.timing import timer

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _run_graph_turn(
    graph: CompiledStateGraph[GraphState, Any, Any, Any],
    session_id: str,
    message: str,
    history_text: str,
) -> tuple[dict[str, Any], float]:
    """Invoke the compiled graph for one turn and time it.

    Returns:
        A tuple of `(final_state, latency_ms)`. `final_state` is the
        dict-like final state LangGraph returns after running the graph
        to completion (confirmed by direct testing — see
        `tests/integration/test_full_graph_run.py`), indexable with `[...]`
        for every `GraphState` field.
    """
    initial_state = GraphState(
        session_id=session_id, original_question=message, history_text=history_text
    )
    with timer() as t:
        final_state = graph.invoke(initial_state)
    return final_state, t.elapsed_ms


def _sse_event(event: str, data: dict[str, Any]) -> str:
    """Format one Server-Sent Events frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream_answer(
    answer: str, session_id: str, sources: list[dict[str, Any]], metrics: ChatMetrics
) -> AsyncIterator[str]:
    """Yield the already-generated, already-verified answer as incremental
    SSE `token` events, followed by one `done` event carrying citations
    and metrics.

    Args:
        answer: The final, graph-verified answer text.
        session_id: The session this answer belongs to.
        sources: Serialized source citations.
        metrics: Observability metrics for this turn.

    Yields:
        SSE-formatted string frames.
    """
    words = answer.split(" ")
    for index, word in enumerate(words):
        chunk = word if index == 0 else f" {word}"
        yield _sse_event("token", {"content": chunk})
    yield _sse_event(
        "done",
        {
            "session_id": session_id,
            "sources": sources,
            "metrics": metrics.model_dump(),
        },
    )


@router.post("/chat", response_model=None, summary="Ask a question against the knowledge base.")
async def chat(
    request: ChatRequest,
    graph: CompiledStateGraph[GraphState, Any, Any, Any] = Depends(get_graph),
    session_store: SessionStore = Depends(get_session_store),
    memory: ConversationMemory = Depends(get_conversation_memory),
) -> StreamingResponse | ChatResponse:
    """Run one turn of the adaptive RAG graph and return the answer.

    Args:
        request: The chat request (message, optional session ID, stream flag).
        graph: Injected compiled LangGraph.
        session_store: Injected session store.
        memory: Injected conversation memory adapter.

    Returns:
        A `StreamingResponse` (SSE) if `request.stream` is `True`
        (default), otherwise a `ChatResponse` JSON body.
    """
    session_id = session_store.ensure_session(request.session_id)
    history_text = memory.get_history_text(session_id)

    final_state, latency_ms = await run_in_threadpool(
        _run_graph_turn, graph, session_id, request.message, history_text
    )

    answer = final_state["generation"]
    retrieved_documents = final_state["retrieved_documents"]
    citations = build_citations(retrieved_documents)
    source_dicts = [c.to_dict() for c in citations]

    metrics = ChatMetrics(
        latency_ms=round(latency_ms, 2),
        route=final_state["route"],
        retrieval_strategy=final_state["retrieval_strategy"],
        used_web_search=final_state["used_web_search"],
        document_relevance_retries=final_state["retry_count_documents"],
        groundedness_retries=final_state["retry_count_groundedness"],
        context_chunks_used=len(final_state["compressed_documents"]),
        token_usage=final_state["token_usage"],
    )

    memory.record_turn(session_id, request.message, answer, sources=source_dicts)

    if request.stream:
        return StreamingResponse(
            _stream_answer(answer, session_id, source_dicts, metrics),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Session-Id": session_id},
        )

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        sources=[SourceCitationSchema(**d) for d in source_dicts],
        metrics=metrics,
    )


@router.get(
    "/chat/{session_id}/export",
    response_model=SessionExportResponse,
    summary="Export a session's full conversation transcript.",
)
async def export_session(
    session_id: str, memory: ConversationMemory = Depends(get_conversation_memory)
) -> SessionExportResponse:
    """Export a session's conversation as a plain-text transcript.

    Args:
        session_id: The session to export.
        memory: Injected conversation memory adapter.

    Returns:
        The full transcript, suitable for a "download conversation" button.

    Raises:
        SessionNotFoundError: If `session_id` does not exist.
    """
    transcript = memory.format_as_transcript(session_id)
    return SessionExportResponse(session_id=session_id, transcript=transcript)


@router.delete(
    "/chat/{session_id}",
    summary="Clear a session's conversation history.",
)
async def clear_session(
    session_id: str, session_store: SessionStore = Depends(get_session_store)
) -> dict[str, str]:
    """Clear a session's message history (the session itself is kept, so
    the same `session_id` can continue to be used for a fresh conversation).

    Args:
        session_id: The session to clear.
        session_store: Injected session store.

    Returns:
        A confirmation message.

    Raises:
        SessionNotFoundError: If `session_id` does not exist.
    """
    if not session_store.session_exists(session_id):
        raise SessionNotFoundError(
            f"Session '{session_id}' does not exist.", details={"session_id": session_id}
        )
    session_store.clear_session(session_id)
    return {"message": f"Session '{session_id}' cleared.", "session_id": session_id}
