"""Request-ID propagation middleware.

Every incoming request is assigned a unique correlation ID — either
honoring an inbound ``X-Request-ID`` header (useful when this service sits
behind a gateway that already assigns one) or generating a fresh one. The
ID is stashed on `request.state` for handlers/dependencies to read, and is
always echoed back on the response so clients can reference it when
reporting issues.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from backend.config.constants import REQUEST_ID_HEADER
from backend.utils.ids import generate_request_id


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assigns and propagates a unique ID for every HTTP request."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Bind a request ID onto the request and response.

        Args:
            request: The incoming HTTP request.
            call_next: The next handler in the ASGI middleware chain.

        Returns:
            The response produced downstream, with the request ID header
            attached.
        """
        request_id = request.headers.get(REQUEST_ID_HEADER) or generate_request_id()
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
