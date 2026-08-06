"""Domain-specific exception hierarchy for the Adaptive RAG Engine.

Every exception raised intentionally by application code (as opposed to
unexpected third-party/library errors) should derive from :class:`AppException`.
This lets the FastAPI error-handling middleware (see
`backend.api.middleware.error_handler`) translate any application error into
a consistent, well-structured JSON response without special-casing every
call site.

Each subclass declares:
    * ``status_code`` — the HTTP status code the API layer should return.
    * ``error_code``   — a short, machine-readable, stable identifier that
      API consumers (the Streamlit frontend, external clients, tests) can
      branch on without parsing free-text messages.
"""

from __future__ import annotations

from typing import Any


class AppException(Exception):  # noqa: N818 - "Exception" suffix is intentional; see module docstring
    """Base class for all intentionally-raised application errors.

    Attributes:
        message: Human-readable description of what went wrong.
        details: Optional structured context (e.g. validation errors,
            offending field names, file names) useful for debugging and
            for returning richer error payloads to API clients.
        status_code: HTTP status code associated with this error class.
        error_code: Stable, machine-readable error identifier.
    """

    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable description of the failure.
            details: Optional structured context about the failure.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(AppException):
    """Raised when application configuration is missing, malformed, or
    otherwise unusable (e.g. a required environment variable is absent)."""

    status_code = 500
    error_code = "configuration_error"


class DocumentProcessingError(AppException):
    """Raised when an uploaded document cannot be parsed or processed."""

    status_code = 422
    error_code = "document_processing_error"


class UnsupportedFileTypeError(DocumentProcessingError):
    """Raised when an uploaded file's extension is not in the allow-list."""

    status_code = 415
    error_code = "unsupported_file_type"


class FileTooLargeError(DocumentProcessingError):
    """Raised when an uploaded file exceeds the configured size limit."""

    status_code = 413
    error_code = "file_too_large"


class CorruptedFileError(DocumentProcessingError):
    """Raised when a file matches an allowed extension but cannot actually
    be parsed (e.g. a corrupted or password-protected PDF)."""

    status_code = 422
    error_code = "corrupted_file"


class RetrievalError(AppException):
    """Raised when the retrieval subsystem (vector store, BM25 index,
    reranker) fails in a way that prevents answering the query."""

    status_code = 502
    error_code = "retrieval_error"


class IndexNotFoundError(AppException):
    """Raised when an operation requires a persisted index (FAISS/BM25)
    that has not yet been built."""

    status_code = 404
    error_code = "index_not_found"


class DocumentNotFoundError(AppException):
    """Raised when a requested document ID does not exist in the document
    registry."""

    status_code = 404
    error_code = "document_not_found"


class GenerationError(AppException):
    """Raised when the LLM generation step fails irrecoverably."""

    status_code = 502
    error_code = "generation_error"


class ExternalServiceError(AppException):
    """Raised when a third-party service (Groq, Tavily, HuggingFace Hub)
    returns an error or is unreachable."""

    status_code = 503
    error_code = "external_service_error"


class SessionNotFoundError(AppException):
    """Raised when a chat session ID does not exist in the session store."""

    status_code = 404
    error_code = "session_not_found"
