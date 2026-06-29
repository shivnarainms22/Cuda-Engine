"""Tests for OpenAICompatibleClient — generic any-provider adapter (no SDK needed)."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from cuda_engine.services.llm.base import LLMResponse
from cuda_engine.services.llm.openai_compatible import OpenAICompatibleClient


def _fake_client(resp: Any) -> SimpleNamespace:
    recorded: list[dict] = []

    def create(**kwargs: Any) -> Any:
        recorded.append(kwargs)
        return resp

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        _recorded=recorded,
    )


def _text_resp(text: str = "hi") -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2, prompt_tokens_details=None),
    )


def test_provider_name_propagates_to_response() -> None:
    client = OpenAICompatibleClient(
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        provider_name="openrouter",
        client=_fake_client(_text_resp()),
    )
    resp = client.complete(system=[], messages=[], model="some/model")
    assert isinstance(resp, LLMResponse)
    assert resp.provider == "openrouter"
    assert resp.text == "hi"


def test_capabilities_conservative_no_prompt_caching() -> None:
    client = OpenAICompatibleClient(
        base_url="https://x/v1",
        api_key_env="X_KEY",
        provider_name="x",
        client=_fake_client(_text_resp()),
    )
    caps = client.capabilities
    assert caps.provider == "x"
    assert caps.prompt_caching is False  # unknown backend → conservative
    assert caps.tool_use is True


def test_cache_control_input_records_degraded() -> None:
    client = OpenAICompatibleClient(
        base_url="https://x/v1",
        api_key_env="X_KEY",
        provider_name="x",
        client=_fake_client(_text_resp()),
    )
    resp = client.complete(
        system=[{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
        messages=[],
        model="m",
    )
    assert resp.degraded == ["prompt_caching"]


def test_reuses_openai_translation_for_messages() -> None:
    fake = _fake_client(_text_resp())
    client = OpenAICompatibleClient(
        base_url="https://x/v1", api_key_env="X_KEY", provider_name="x", client=fake
    )
    client.complete(system=[], messages=[{"role": "user", "content": "yo"}], model="m")
    sent = fake._recorded[0]["messages"]
    assert sent == [{"role": "user", "content": "yo"}]
