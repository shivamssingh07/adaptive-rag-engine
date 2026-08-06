"""Centralized exception handling.

Registers FastAPI exception handlers that translate every error — whether
an intentional `AppException` subclass, a request validation failure, or an
unexpected/unhandled exception — into the same `ErrorResponse` envelope
(see `backend.api.schemas.common.ErrorResponse`), so API consumers never
have to special-case error shapes by endpoint.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.api.schemas.common import ErrorResponse
from backend.core.exceptions import AppException

logger = logging.getLogger(__name__)


def _request_id(request: Request) -> str | None:
    """Best-effort extraction of the request's correlation ID.

    The `RequestIDMiddleware` runs before route handlers but exception
    handlers can, in rare edge cases (e.g. a malformed request that fails
    before middleware fully runs), execute without `request.state`
    populated — this helper defends against that.
    """
    return getattr(request.state, "request_id", None)


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all global exception handlers to the FastAPI application.

    Args:
        app: The FastAPI application instance to register handlers on.
    """

    @app.exception_handler(AppException)
    async def handle_app_exception(request: Request, exc: AppException) -> JSONResponse:
        """Handle every intentional domain error raised by application code."""
        logger.warning(
            "Application error: %s (%s)",
            exc.message,
            exc.error_code,
            extra={
                "request_id": _request_id(request),
                "error_code": exc.error_code,
                "details": exc.details,
            },
        )
        body = ErrorResponse(
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details or None,
            request_id=_request_id(request),
        )
        return JSONResponse(status_code=exc.status_code, content=body.model_dump(mode="json"))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle Pydantic/FastAPI request validation failures (422)."""
        logger.info(
            "Request validation failed: %s",
            exc.errors(),
            extra={"request_id": _request_id(request)},
        )
        body = ErrorResponse(
            error_code="validation_error",
            message="The request did not pass validation.",
            details={"errors": exc.errors()},
            request_id=_request_id(request),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=body.model_dump(mode="json"),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Handle plain Starlette/FastAPI HTTPExceptions (e.g. 404 on an
        unregistered route) with the same error envelope as everything
        else."""
        body = ErrorResponse(
            error_code="http_error",
            message=str(exc.detail),
            request_id=_request_id(request),
        )
        return JSONResponse(status_code=exc.status_code, content=body.model_dump(mode="json"))

    @app.exception_handler(Exception)
    async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all for any exception not otherwise handled above.

        Logs the full traceback server-side (`exc_info=True`) but never
        leaks internal exception details to the client — only a generic
        message and the request ID the client can use to reference this
        specific failure when reporting it.
        """
        logger.error(
            "Unhandled exception while processing request",
            exc_info=True,
            extra={"request_id": _request_id(request)},
        )
        body = ErrorResponse(
            error_code="internal_error",
            message="An unexpected error occurred. Please try again or contact support "
            "with the request ID below.",
            request_id=_request_id(request),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=body.model_dump(mode="json"),
        )
