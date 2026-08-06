"""SQLite-backed session store.

Chat sessions and their message history are persisted to a local SQLite
database (`settings.session_db_path`) rather than kept only in an
in-memory dict, so conversations survive an application restart — at zero
infrastructure cost, consistent with the project's zero-paid-dependency
goal.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.config.constants import DEFAULT_SESSION_TTL_SECONDS
from backend.config.settings import Settings, get_settings
from backend.core.exceptions import SessionNotFoundError
from backend.utils.ids import generate_session_id

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChatTurn:
    """A single message in a conversation."""

    role: str  # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    sources: list[dict[str, Any]] | None = None


class SessionStore:
    """Thread-safe, disk-persisted store for chat session conversation
    history."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the store and create its schema if it doesn't exist.

        Args:
            settings: Application settings. Defaults to the process-wide
                settings singleton.
        """
        self._settings = settings or get_settings()
        self._lock = threading.RLock()
        self._db_path: Path = self._settings.session_db_path
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path, check_same_thread=False)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _init_schema(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources TEXT,
                    timestamp REAL NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)"
            )
            connection.commit()
        logger.info("Session store schema ready at %s", self._db_path)

    def create_session(self) -> str:
        """Create a new, empty session.

        Returns:
            The newly generated session ID.
        """
        session_id = generate_session_id()
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions (session_id, created_at, updated_at) VALUES (?, ?, ?)",
                (session_id, now, now),
            )
            connection.commit()
        logger.info("Created new session %s", session_id)
        return session_id

    def session_exists(self, session_id: str) -> bool:
        """Check whether a session ID exists.

        Args:
            session_id: The session ID to check.

        Returns:
            `True` if the session exists.
        """
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return row is not None

    def ensure_session(self, session_id: str | None) -> str:
        """Return a valid session ID, creating a new session if `session_id`
        is `None` or refers to a session that no longer exists.

        Args:
            session_id: A client-supplied session ID, or `None` for a new
                conversation.

        Returns:
            A valid, existing session ID.
        """
        if session_id and self.session_exists(session_id):
            return session_id
        return self.create_session()

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: list[dict[str, Any]] | None = None,
    ) -> None:
        """Append a message to a session's history.

        Args:
            session_id: The session to append to.
            role: `"user"` or `"assistant"`.
            content: The message text.
            sources: Optional source citations to store alongside an
                assistant message.

        Raises:
            SessionNotFoundError: If `session_id` does not exist.
        """
        if not self.session_exists(session_id):
            raise SessionNotFoundError(
                f"Session '{session_id}' does not exist.", details={"session_id": session_id}
            )
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO messages (session_id, role, content, sources, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, json.dumps(sources) if sources else None, now),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, session_id)
            )
            connection.commit()

    def get_history(self, session_id: str, limit: int | None = None) -> list[ChatTurn]:
        """Return a session's message history in chronological order.

        Args:
            session_id: The session to read.
            limit: If given, return only the `limit` most recent messages
                (still chronologically ordered in the result).

        Returns:
            The session's `ChatTurn` history.

        Raises:
            SessionNotFoundError: If `session_id` does not exist.
        """
        if not self.session_exists(session_id):
            raise SessionNotFoundError(
                f"Session '{session_id}' does not exist.", details={"session_id": session_id}
            )

        if limit is not None:
            query = (
                "SELECT role, content, sources, timestamp FROM messages "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?"
            )
            params: tuple[Any, ...] = (session_id, limit)
        else:
            query = (
                "SELECT role, content, sources, timestamp FROM messages "
                "WHERE session_id = ? ORDER BY id ASC"
            )
            params = (session_id,)

        with self._lock, self._connect() as connection:
            rows = connection.execute(query, params).fetchall()

        turns = [
            ChatTurn(
                role=row[0],
                content=row[1],
                sources=json.loads(row[2]) if row[2] else None,
                timestamp=row[3],
            )
            for row in rows
        ]
        if limit is not None:
            turns.reverse()
        return turns

    def clear_session(self, session_id: str) -> None:
        """Delete all messages in a session, keeping the session itself.

        Args:
            session_id: The session to clear.

        Raises:
            SessionNotFoundError: If `session_id` does not exist.
        """
        if not self.session_exists(session_id):
            raise SessionNotFoundError(
                f"Session '{session_id}' does not exist.", details={"session_id": session_id}
            )
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            connection.commit()
        logger.info("Cleared conversation history for session %s", session_id)

    def delete_session(self, session_id: str) -> None:
        """Delete a session and all of its messages.

        Args:
            session_id: The session to delete.
        """
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            connection.commit()
        logger.info("Deleted session %s", session_id)

    def purge_expired_sessions(self, ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS) -> int:
        """Delete sessions that haven't been touched in `ttl_seconds`.

        Args:
            ttl_seconds: Idle time after which a session is considered
                expired. Defaults to `DEFAULT_SESSION_TTL_SECONDS` (7 days).

        Returns:
            The number of sessions deleted.
        """
        cutoff = time.time() - ttl_seconds
        with self._lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM sessions WHERE updated_at < ?", (cutoff,))
            connection.commit()
            deleted = cursor.rowcount
        if deleted:
            logger.info("Purged %d expired session(s).", deleted)
        return deleted

    @property
    def session_count(self) -> int:
        """Total number of active (non-expired-purged) sessions, used by
        the `/metrics` endpoint."""
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM sessions").fetchone()
        return int(row[0]) if row else 0


_store_singleton: SessionStore | None = None
_store_lock = threading.Lock()


def get_session_store() -> SessionStore:
    """Return the process-wide `SessionStore` singleton.

    Returns:
        The shared `SessionStore` instance.
    """
    global _store_singleton
    if _store_singleton is None:
        with _store_lock:
            if _store_singleton is None:
                _store_singleton = SessionStore()
    return _store_singleton
