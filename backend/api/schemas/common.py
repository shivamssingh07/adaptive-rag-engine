"""Common API schemas shared across multiple routes."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class HealthStatus(StrEnum):
    """Overall health classification returned by `GET /health`."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentStatus(BaseModel):
    """Configuration/availability status of a single backend component."""

    name: str = Field(..., description="Component identifier, e.g. 'groq_llm'.")
    configured: bool = Field(..., description="Whether this component is usable.")
    detail: str | None = Field(default=None, description="Human-readable extra context, if any.")


class HealthResponse(BaseModel):
    """Response body for `GET /health`."""

    status: HealthStatus
    app_name: str
    version: str
    environment: str
    uptime_seconds: float = Field(..., ge=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    components: list[ComponentStatus] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Standard error envelope returned by every failure response.

    Every `AppException` subclass (see `backend.core.exceptions`) and every
    unhandled exception raised inside a request is translated into this
    shape by `backend.api.middleware.error_handler`, so API consumers only
    ever need to handle one error format.
    """

    error_code: str = Field(..., description="Stable, machine-readable error identifier.")
    message: str = Field(..., description="Human-readable description of the error.")
    details: dict[str, Any] | None = Field(
        default=None, description="Optional structured context about the error."
    )
    request_id: str | None = Field(
        default=None, description="Correlates this error with server-side logs."
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
