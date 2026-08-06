"""ID generation utilities.

Every identifier generated here is prefixed with a short type tag
(``req_``, ``sess_``, ``doc_``) so that IDs are self-describing when they
show up in logs, error payloads, or API responses — you can tell what kind
of entity an ID refers to without cross-referencing a schema.
"""

from __future__ import annotations

import uuid

from backend.config.constants import DOCUMENT_ID_PREFIX, REQUEST_ID_PREFIX, SESSION_ID_PREFIX


def _new_id(prefix: str) -> str:
    """Build a prefixed, collision-resistant identifier.

    Args:
        prefix: Short type tag prepended to the generated UUID hex.

    Returns:
        A string of the form ``"{prefix}_{32-char-hex}"``.
    """
    return f"{prefix}_{uuid.uuid4().hex}"


def generate_request_id() -> str:
    """Generate a new unique request identifier (e.g. for HTTP request
    correlation)."""
    return _new_id(REQUEST_ID_PREFIX)


def generate_session_id() -> str:
    """Generate a new unique chat session identifier."""
    return _new_id(SESSION_ID_PREFIX)


def generate_document_id() -> str:
    """Generate a new unique identifier for an uploaded source document."""
    return _new_id(DOCUMENT_ID_PREFIX)


def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """Generate a deterministic identifier for a chunk of a document.

    Deterministic (rather than random) so that re-processing the same
    document with the same splitter configuration produces the same chunk
    IDs, which makes de-duplication and idempotent re-indexing possible.

    Args:
        document_id: The parent document's identifier.
        chunk_index: The zero-based position of this chunk within the
            document.

    Returns:
        A string of the form ``"{document_id}_chunk_{chunk_index}"``.
    """
    return f"{document_id}_chunk_{chunk_index}"
