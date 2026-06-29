from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from cuda_engine.services.llm.capabilities import ProviderCapabilities


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class LLMResponse(BaseModel):
    text: str
    model: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read_tokens: int = 0
    latency_seconds: float = 0.0
    provider: str = ""
    degraded: list[str] = Field(default_factory=list)


class LLMClient(ABC):
    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError
