from typing import Any

from cuda_engine.services.llm.base import LLMClient, LLMResponse, ToolSpec
from cuda_engine.services.llm.capabilities import ProviderCapabilities

_MOCK_CAPABILITIES = ProviderCapabilities(provider="mock")


class MockLLMClient(LLMClient):
    def __init__(
        self,
        responses: list[str | LLMResponse],
        capabilities: ProviderCapabilities | None = None,
    ) -> None:
        self._responses = list(responses)
        self._capabilities = capabilities if capabilities is not None else _MOCK_CAPABILITIES
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

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
        self.calls.append(
            {
                "system": system,
                "messages": messages,
                "tools": tools,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if not self._responses:
            raise RuntimeError("MockLLMClient: no canned responses left")
        next_response = self._responses.pop(0)
        self.call_count += 1
        if isinstance(next_response, LLMResponse):
            return next_response
        return LLMResponse(
            text=next_response, model="mock", provider="mock", tokens_in=10, tokens_out=10
        )
