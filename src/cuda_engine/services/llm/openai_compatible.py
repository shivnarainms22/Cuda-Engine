"""Generic adapter for any OpenAI-API-compatible endpoint.

This is the "flexible with any provider" piece: point it at any ``base_url``
that speaks the OpenAI chat-completions protocol (OpenRouter, Together, Groq,
DeepSeek, vLLM/local models, …). It reuses ``OpenAIClient``'s translation and
request handling, differing only in the endpoint, the reported provider name,
and conservative capabilities (an unknown backend is assumed NOT to honour
explicit prompt-cache hints).
"""
from __future__ import annotations

from typing import Any

from cuda_engine.services.llm.capabilities import ProviderCapabilities
from cuda_engine.services.llm.openai_client import OpenAIClient


class OpenAICompatibleClient(OpenAIClient):
    """OpenAIClient pointed at an arbitrary OpenAI-compatible ``base_url``."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key_env: str,
        provider_name: str,
        client: Any = None,
    ) -> None:
        super().__init__(
            client=client,
            api_key_env=api_key_env,
            base_url=base_url,
            provider=provider_name,
            capabilities=ProviderCapabilities(
                provider=provider_name,
                prompt_caching=False,  # unknown backend → conservative
                tool_use=True,
            ),
        )
