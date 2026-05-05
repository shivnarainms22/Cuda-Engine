import os
import time
from collections.abc import Callable
from typing import Any, cast

import anthropic

from cuda_engine.config import SynthesisConfig
from cuda_engine.services.llm.base import LLMClient, LLMResponse, ToolSpec


class AnthropicClient(LLMClient):
    def __init__(self, cfg: SynthesisConfig | None = None) -> None:
        self.cfg = cfg or SynthesisConfig()
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

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
        started_at = time.time()
        create = cast(Callable[..., Any], self.client.messages.create)
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": messages,
        }
        if tools is not None:
            request["tools"] = [tool.model_dump(mode="json") for tool in tools]
        response = create(**request)
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in getattr(response, "content", []):
            block_type = _read_attr(block, "type")
            if block_type == "text":
                text_parts.append(str(_read_attr(block, "text") or ""))
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "name": _read_attr(block, "name"),
                        "input": _read_attr(block, "input") or {},
                    }
                )

        usage = getattr(response, "usage", None)
        return LLMResponse(
            text="\n".join(part for part in text_parts if part),
            tool_calls=tool_calls,
            model=str(getattr(response, "model", model)),
            tokens_in=int(getattr(usage, "input_tokens", 0) if usage is not None else 0),
            tokens_out=int(getattr(usage, "output_tokens", 0) if usage is not None else 0),
            cache_read_tokens=int(
                getattr(usage, "cache_read_input_tokens", 0) if usage is not None else 0
            ),
            latency_seconds=time.time() - started_at,
        )


def _read_attr(obj: object, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
