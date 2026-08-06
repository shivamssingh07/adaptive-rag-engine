"""Structured access-logging middleware.

Logs one structured record per HTTP request, after it completes, including
method, path, status code, latency, and the request's correlation ID (so it
can be joined against the deeper per-request logs emitted by the graph
nodes during `/chat` processing).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from backend.config.constants import PROCESS_TIME_HEADER
from backend.utils.timing import timer

logger = logging.getLogger("backend.access")


class LoggingMiddleware(BaseHTTPMiddleware):
    """Logs a structured access record for every completed HTTP request."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Time the downstream handler and emit an access-log record.

        Args:
            request: The incoming HTTP request.
            call_next: The next handler in the ASGI middleware chain.

        Returns:
            The response produced downstream, with a
            ``X-Process-Time-Ms`` header attached.
        """
        request_id = getattr(request.state, "request_id", "-")

        with timer() as t:
            response = await call_next(request)

        response.headers[PROCESS_TIME_HEADER] = f"{t.elapsed_ms:.2f}"

        logger.info(
            "%s %s -> %s (%.2fms)",
            request.method,
            request.url.path,
            response.status_code,
            t.elapsed_ms,
            extra={
                "request_id": request_id,
                "http_method": request.method,
                "http_path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(t.elapsed_ms, 2),
                "client_host": request.client.host if request.client else None,
            },
        )
        return response
