"""Gemini native LLM client adapter.

The real google-genai SDK is imported lazily (inside __init__) only when no
client is injected, so tests can inject a fake without the SDK installed.
"""
from __future__ import annotations

import os
import time
from typing import Any

from cuda_engine.services.llm.base import LLMClient, LLMResponse, ToolSpec
from cuda_engine.services.llm.capabilities import ProviderCapabilities
from cuda_engine.services.llm.translate import (
    from_gemini_response,
    has_cache_control,
    to_gemini,
)

_GEMINI_CAPABILITIES = ProviderCapabilities(
    provider="gemini", prompt_caching=True, tool_use=True
)


class GeminiClient(LLMClient):
    """Adapts LLMClient to Google's Gemini generate_content API."""

    def __init__(
        self, *, client: Any = None, api_key_env: str = "GEMINI_API_KEY"
    ) -> None:
        if client is None:
            from google import genai  # type: ignore[import-untyped, attr-defined]

            self._client = genai.Client(api_key=os.environ.get(api_key_env))
        else:
            self._client = client

    @property
    def capabilities(self) -> ProviderCapabilities:
        return _GEMINI_CAPABILITIES

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
        degraded = ["prompt_caching"] if has_cache_control(system, messages) else []
        payload = to_gemini(system, messages, tools)

        config: dict[str, Any] = {"max_output_tokens": max_tokens}
        if payload["system_instruction"]:
            config["system_instruction"] = payload["system_instruction"]
        if temperature is not None:
            config["temperature"] = temperature
        if payload["tools"]:
            config["tools"] = payload["tools"]

        started_at = time.time()
        resp = self._client.models.generate_content(
            model=model,
            contents=payload["contents"],
            config=config,
        )
        parsed = from_gemini_response(resp)

        return LLMResponse(
            text=parsed["text"],
            tool_calls=parsed["tool_calls"],
            model=model,
            provider="gemini",
            tokens_in=parsed["tokens_in"],
            tokens_out=parsed["tokens_out"],
            cache_read_tokens=parsed["cache_read_tokens"],
            latency_seconds=time.time() - started_at,
            degraded=degraded,
        )
