"""Application-wide constants.

Anything a deployer might reasonably want to override at runtime belongs in
`backend.config.settings.Settings` (backed by environment variables).
Anything that is a fixed property of *this codebase* — API prefixes, header
names, graph node identifiers, default file extensions the loaders know how
to parse — belongs here instead.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
API_V1_PREFIX: Final[str] = "/api/v1"
REQUEST_ID_HEADER: Final[str] = "X-Request-ID"
PROCESS_TIME_HEADER: Final[str] = "X-Process-Time-Ms"

# --------------------------------------------------------------------------
# Document ingestion
# --------------------------------------------------------------------------
SUPPORTED_FILE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".pdf", ".docx", ".txt", ".md", ".csv"}
)

EXTENSION_MIME_TYPES: Final[dict[str, str]] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
}

# --------------------------------------------------------------------------
# Retrieval defaults (used as fallback values, not the source of truth —
# runtime-tunable values live in Settings)
# --------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE: Final[int] = 1000
DEFAULT_CHUNK_OVERLAP: Final[int] = 200
DEFAULT_TOP_K_RETRIEVAL: Final[int] = 10
DEFAULT_TOP_K_RERANK: Final[int] = 4

# --------------------------------------------------------------------------
# Adaptive graph node / route identifiers
#
# Using an Enum (rather than bare strings scattered through the graph
# module) means a typo in a routing decision fails fast with an
# AttributeError at development time instead of silently mis-routing a
# query in production.
# --------------------------------------------------------------------------


class GraphRoute(StrEnum):
    """Possible routing decisions made by the router node."""

    VECTORSTORE = "vectorstore"
    WEB_SEARCH = "web_search"
    DIRECT_ANSWER = "direct_answer"


class GraphNode(StrEnum):
    """Canonical names of every node registered in the LangGraph state
    machine. Referenced when wiring edges in `backend.core.graph.builder`."""

    ROUTE_QUESTION = "route_question"
    RETRIEVE = "retrieve"
    GRADE_DOCUMENTS = "grade_documents"
    REWRITE_QUERY = "rewrite_query"
    WEB_SEARCH = "web_search"
    GENERATE = "generate"
    GENERATE_DIRECT = "generate_direct"
    GRADE_GENERATION = "grade_generation"


class RelevanceGrade(StrEnum):
    """Binary relevance grade assigned to a retrieved document."""

    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"


class GroundednessGrade(StrEnum):
    """Binary groundedness grade assigned to a generated answer."""

    GROUNDED = "grounded"
    NOT_GROUNDED = "not_grounded"


# --------------------------------------------------------------------------
# Session / memory
# --------------------------------------------------------------------------
DEFAULT_SESSION_TTL_SECONDS: Final[int] = 60 * 60 * 24 * 7  # 7 days
SESSION_ID_PREFIX: Final[str] = "sess"
REQUEST_ID_PREFIX: Final[str] = "req"
DOCUMENT_ID_PREFIX: Final[str] = "doc"

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
LOG_FILE_NAME: Final[str] = "app.log"
LOG_FILE_MAX_BYTES: Final[int] = 10 * 1024 * 1024  # 10 MB
LOG_FILE_BACKUP_COUNT: Final[int] = 5
