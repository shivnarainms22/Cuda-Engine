"""OpenAI native LLM client adapter.

The real openai SDK is imported lazily (inside __init__) only when no client
is injected, so tests can inject a fake without the SDK installed.
"""
from __future__ import annotations

import os
import time
from typing import Any

from cuda_engine.services.llm.base import LLMClient, LLMResponse, ToolSpec
from cuda_engine.services.llm.capabilities import ProviderCapabilities
from cuda_engine.services.llm.translate import (
    from_openai_response,
    to_openai_messages,
    to_openai_tools,
)

_OPENAI_CAPABILITIES = ProviderCapabilities(
    provider="openai", prompt_caching=True, tool_use=True
)


class OpenAIClient(LLMClient):
    """Adapts LLMClient to OpenAI's chat completions API."""

    def __init__(self, *, client: Any = None, api_key_env: str = "OPENAI_API_KEY") -> None:
        if client is None:
            import openai  # lazy — only when not injected

            self._client = openai.OpenAI(api_key=os.environ.get(api_key_env))
        else:
            self._client = client

    @property
    def capabilities(self) -> ProviderCapabilities:
        return _OPENAI_CAPABILITIES

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
        degraded = ["prompt_caching"] if _has_cache_control(system, messages) else []

        started_at = time.time()
        oai_messages = to_openai_messages(system, messages)
        oai_tools = to_openai_tools(tools)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": oai_messages,
            "max_tokens": max_tokens,
        }
        if oai_tools is not None:
            kwargs["tools"] = oai_tools
        if temperature is not None:
            kwargs["temperature"] = temperature

        resp = self._client.chat.completions.create(**kwargs)
        parsed = from_openai_response(resp)

        return LLMResponse(
            text=parsed["text"],
            tool_calls=parsed["tool_calls"],
            model=model,
            provider="openai",
            tokens_in=parsed["tokens_in"],
            tokens_out=parsed["tokens_out"],
            cache_read_tokens=parsed["cache_read_tokens"],
            latency_seconds=time.time() - started_at,
            degraded=degraded,
        )


def _has_cache_control(
    system: list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> bool:
    """Return True if any block in system or messages carries a cache_control key."""
    for block in system:
        if "cache_control" in block:
            return True
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if "cache_control" in block:
                    return True
    return False
