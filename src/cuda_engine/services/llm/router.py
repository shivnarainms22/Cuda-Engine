from __future__ import annotations

from typing import Any

from cuda_engine.services.llm.base import LLMClient, LLMResponse, ToolSpec
from cuda_engine.services.llm.capabilities import ProviderCapabilities


def parse_model_id(model: str) -> tuple[str, str]:
    """Split ``"provider:model"`` into ``(provider, model)``.

    A bare model string with no ``:`` defaults to the anthropic provider.
    Only the first ``:`` is treated as the separator.
    """
    if ":" in model:
        provider, bare = model.split(":", 1)
        return provider, bare
    return "anthropic", model


class LLMRouter(LLMClient):
    """Routes ``complete`` calls to the appropriate provider based on the model id prefix."""

    def __init__(self, providers: dict[str, LLMClient]) -> None:
        self._providers = providers

    @property
    def capabilities(self) -> ProviderCapabilities:
        if "anthropic" in self._providers:
            return self._providers["anthropic"].capabilities
        return next(iter(self._providers.values())).capabilities

    def complete(
        self,
        *,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        model: str,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> LLMResponse:
        provider_name, bare_model = parse_model_id(model)
        provider = self._providers[provider_name]  # raises KeyError for unknown provider
        return provider.complete(
            system=system,
            messages=messages,
            tools=tools,
            model=bare_model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
