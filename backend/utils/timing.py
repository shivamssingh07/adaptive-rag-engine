"""Latency measurement utilities.

Used throughout the RAG pipeline (retrieval, reranking, generation) and the
API layer to report per-request latency to clients (see
`ChatResponse.latency_ms` in `backend.api.schemas.chat`, added in a later
phase) and to structured logs.
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")

_default_logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TimingResult:
    """Mutable container for an elapsed-time measurement.

    Populated in place by :func:`timer` once its `with` block exits, since
    the elapsed time can't be known until the block completes.
    """

    elapsed_ms: float = 0.0


@contextmanager
def timer() -> Iterator[TimingResult]:
    """Context manager that measures wall-clock elapsed time.

    Example:
        >>> with timer() as t:
        ...     do_expensive_work()
        >>> print(t.elapsed_ms)

    Yields:
        A :class:`TimingResult` whose ``elapsed_ms`` field is populated
        once the ``with`` block exits (including if it exits via an
        exception).
    """
    result = TimingResult()
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_ms = (time.perf_counter() - start) * 1000


def timed_async(
    logger: logging.Logger | None = None,
) -> Callable[[Callable[_P, Awaitable[_T]]], Callable[_P, Awaitable[_T]]]:
    """Decorator that logs the latency of an async function call.

    Args:
        logger: Logger to emit the timing record to. Defaults to a logger
            named after this module if not provided.

    Returns:
        A decorator suitable for wrapping any `async def` function.

    Example:
        >>> @timed_async()
        ... async def retrieve(query: str) -> list[str]:
        ...     ...
    """

    def decorator(func: Callable[_P, Awaitable[_T]]) -> Callable[_P, Awaitable[_T]]:
        @functools.wraps(func)
        async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            active_logger = logger or _default_logger
            with timer() as t:
                result = await func(*args, **kwargs)
            active_logger.debug(
                "%s completed in %.2fms",
                func.__qualname__,
                t.elapsed_ms,
            )
            return result

        return wrapper

    return decorator
