from typing import Any

from cuda_engine.config import SynthesisConfig
from cuda_engine.services.llm.base import LLMClient, LLMResponse, ToolSpec


class AnthropicClient(LLMClient):
    def __init__(self, cfg: SynthesisConfig | None = None) -> None:
        self.cfg = cfg

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
        raise NotImplementedError("AnthropicClient lands in M1")
