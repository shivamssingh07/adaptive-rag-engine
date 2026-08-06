"""Abstract interface for LLM providers.

Defined as a `Protocol` (structural typing) rather than an ABC so that any
object exposing a compatible `get_llm` method satisfies it — useful for
tests that pass in a lightweight fake without inheriting from a real base
class.
"""

from __future__ import annotations

from typing import Protocol

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable


class LLMProvider(Protocol):
    """Structural interface every LLM provider implementation must satisfy."""

    def get_llm(
        self, temperature: float | None = None
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Return a chat model instance, optionally overriding temperature.

        Args:
            temperature: Sampling temperature to use for this call site. If
                ``None``, the provider's configured default is used.

        Returns:
            A `Runnable` (a `BaseChatModel` or a `.bind()`-wrapped variant
            of one) ready to invoke or use in a chain.
        """
        ...
