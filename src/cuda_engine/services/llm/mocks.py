from typing import Any

from cuda_engine.services.llm.base import LLMClient, LLMResponse, ToolSpec


class MockLLMClient(LLMClient):
    def __init__(self, responses: list[str | LLMResponse]) -> None:
        self._responses = list(responses)
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
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
        return LLMResponse(text=next_response, model="mock", tokens_in=10, tokens_out=10)
