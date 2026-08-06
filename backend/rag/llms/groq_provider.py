"""Groq LLM provider.

Wraps `langchain_groq.ChatGroq` behind a thread-safe, lazily-initialized
singleton. The underlying HTTP client is only constructed on first use
(not at import time), and every subsequent call reuses the same client —
per-call temperature overrides use LangChain's `.bind()` rather than
constructing a new client, so the singleton property is preserved even
when different graph nodes need different sampling temperatures (e.g. a
grading node wants temperature 0, generation might want a bit more).
"""

from __future__ import annotations

import logging
import threading

from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq

from backend.config.settings import Settings, get_settings
from backend.core.exceptions import ExternalServiceError

logger = logging.getLogger(__name__)


class GroqLLMProvider:
    """Thread-safe, lazily-initialized wrapper around a Groq chat model."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the provider.

        Note this does NOT construct the underlying `ChatGroq` client —
        that happens lazily on first call to :meth:`get_client`.

        Args:
            settings: Application settings. Defaults to the process-wide
                settings singleton.
        """
        self._settings = settings or get_settings()
        self._client: BaseChatModel | None = None
        self._lock = threading.Lock()

    def get_client(self) -> BaseChatModel:
        """Return the underlying Groq chat model, constructing it on first
        call and caching it for the lifetime of the process.

        Uses double-checked locking so concurrent requests during a cold
        start don't each construct (and discard) their own client.

        Returns:
            The cached `ChatGroq` instance.

        Raises:
            ExternalServiceError: If the Groq client cannot be constructed
                (e.g. malformed credentials).
        """
        if self._client is None:
            with self._lock:
                if self._client is None:
                    logger.info(
                        "Initializing Groq LLM client (model=%s, temperature=%s)",
                        self._settings.groq_model,
                        self._settings.groq_temperature,
                    )
                    try:
                        self._client = ChatGroq(
                            api_key=self._settings.groq_api_key,
                            model=self._settings.groq_model,
                            temperature=self._settings.groq_temperature,
                        )
                    except Exception as exc:  # noqa: BLE001 - wrapped as a domain error
                        raise ExternalServiceError(
                            f"Failed to initialize Groq LLM client: {exc}",
                            details={"model": self._settings.groq_model},
                        ) from exc
                    logger.info("Groq LLM client initialized successfully.")
        return self._client

    def get_llm(
        self, temperature: float | None = None
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Return a chat model ready to invoke, optionally at a different
        sampling temperature than the configured default.

        Args:
            temperature: Overrides the configured default temperature for
                this call site only (via `.bind()`, not re-initialization).

        Returns:
            A `Runnable` — either the cached `BaseChatModel` client
            directly, or a `.bind()`-wrapped variant with the overridden
            temperature (still invocable identically).
        """
        client = self.get_client()
        if temperature is not None and temperature != self._settings.groq_temperature:
            return client.bind(temperature=temperature)
        return client


_provider_singleton: GroqLLMProvider | None = None
_provider_lock = threading.Lock()


def get_groq_provider() -> GroqLLMProvider:
    """Return the process-wide `GroqLLMProvider` singleton.

    Returns:
        The shared `GroqLLMProvider` instance.
    """
    global _provider_singleton
    if _provider_singleton is None:
        with _provider_lock:
            if _provider_singleton is None:
                _provider_singleton = GroqLLMProvider()
    return _provider_singleton
