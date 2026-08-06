"""Conversation memory adapter.

Bridges the persistence layer (`SessionStore`) and the LLM-facing world:
exposes a session's history as LangChain `BaseMessage` objects (ready to
inject into a prompt or the LangGraph state built in Phase 6), and as a
plain-text transcript (for the Streamlit "download conversation" feature).
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from backend.rag.memory.session_store import SessionStore, get_session_store


class ConversationMemory:
    """Adapts SQLite-backed session history into LangChain message objects."""

    def __init__(self, session_store: SessionStore | None = None, max_turns: int = 10) -> None:
        """Args:
        session_store: Backing store. Defaults to the process-wide
            singleton.
        max_turns: Maximum number of past (user, assistant) turn pairs to
            include when building message history for a prompt — keeps
            token usage bounded on long-running conversations.
        """
        self._store = session_store or get_session_store()
        self._max_turns = max_turns

    def get_messages(self, session_id: str) -> list[BaseMessage]:
        """Return the session's recent history as LangChain messages.

        Args:
            session_id: The session to read.

        Returns:
            A chronologically-ordered list of `HumanMessage`/`AIMessage`
            objects, capped at `max_turns` turn pairs.

        Raises:
            SessionNotFoundError: If `session_id` does not exist.
        """
        turns = self._store.get_history(session_id, limit=self._max_turns * 2)
        messages: list[BaseMessage] = []
        for turn in turns:
            if turn.role == "user":
                messages.append(HumanMessage(content=turn.content))
            elif turn.role == "assistant":
                messages.append(AIMessage(content=turn.content))
        return messages

    def get_history_text(self, session_id: str) -> str:
        """Return the session's recent history as a compact text block,
        suitable for injection into `QueryRewriter`'s `history` argument.

        Args:
            session_id: The session to read.

        Returns:
            A newline-joined `"User: ...\\nAssistant: ..."` transcript of
            the most recent turns, or an empty string for a new session.
        """
        turns = self._store.get_history(session_id, limit=self._max_turns * 2)
        if not turns:
            return ""
        lines = [
            f"{'User' if turn.role == 'user' else 'Assistant'}: {turn.content}" for turn in turns
        ]
        return "\n".join(lines)

    def record_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        sources: list[dict[str, Any]] | None = None,
    ) -> None:
        """Persist one (user, assistant) turn pair to the session.

        Args:
            session_id: The session to append to.
            user_message: The user's question, as asked.
            assistant_message: The generated answer.
            sources: Optional source citations to store alongside the
                assistant's message.

        Raises:
            SessionNotFoundError: If `session_id` does not exist.
        """
        self._store.add_message(session_id, "user", user_message)
        self._store.add_message(session_id, "assistant", assistant_message, sources=sources)

    def format_as_transcript(self, session_id: str) -> str:
        """Return the *entire* session history as a human-readable
        transcript, e.g. for a "download conversation" button.

        Args:
            session_id: The session to read.

        Returns:
            A double-newline-separated transcript of every turn in the
            session.

        Raises:
            SessionNotFoundError: If `session_id` does not exist.
        """
        turns = self._store.get_history(session_id)
        lines = [
            f"{'You' if turn.role == 'user' else 'Assistant'}: {turn.content}" for turn in turns
        ]
        return "\n\n".join(lines)
