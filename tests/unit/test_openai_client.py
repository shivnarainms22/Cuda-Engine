"""Tests for OpenAIClient — no real openai SDK needed; client is injected."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from cuda_engine.services.llm.base import LLMResponse, ToolSpec
from cuda_engine.services.llm.openai_client import OpenAIClient

# ---------------------------------------------------------------------------
# Fake OpenAI client helpers
# ---------------------------------------------------------------------------


def _fake_client(resp: Any) -> SimpleNamespace:
    """Fake that mimics openai.OpenAI().chat.completions.create(...)."""
    recorded: list[dict] = []

    def create(**kwargs: Any) -> Any:
        recorded.append(kwargs)
        return resp

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        ),
        _recorded=recorded,
    )
    return client


def _oai_text_resp(text: str = "hi", tokens_in: int = 5, tokens_out: int = 3) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=None)
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=tokens_in,
            completion_tokens=tokens_out,
            prompt_tokens_details=None,
        ),
    )


def _oai_tool_resp(name: str = "fn", args: dict | None = None) -> Any:
    args = args or {"x": 1}
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(
                                name=name, arguments=json.dumps(args)
                            ),
                        )
                    ],
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=4,
            prompt_tokens_details=SimpleNamespace(cached_tokens=2),
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_openai_client_capabilities_provider() -> None:
    client = OpenAIClient(client=_fake_client(_oai_text_resp()))
    assert client.capabilities.provider == "openai"


def test_openai_client_capabilities_supports_tool_use_and_caching() -> None:
    client = OpenAIClient(client=_fake_client(_oai_text_resp()))
    caps = client.capabilities
    assert caps.tool_use is True
    assert caps.prompt_caching is True


def test_openai_client_complete_text_response() -> None:
    fake = _fake_client(_oai_text_resp("Hello from OpenAI", 8, 4))
    oai = OpenAIClient(client=fake)
    resp = oai.complete(system=[], messages=[{"role": "user", "content": "hi"}], model="gpt-4o")
    assert isinstance(resp, LLMResponse)
    assert resp.text == "Hello from OpenAI"
    assert resp.provider == "openai"
    assert resp.model == "gpt-4o"
    assert resp.tokens_in == 8
    assert resp.tokens_out == 4


def test_openai_client_complete_tool_calls_mapped() -> None:
    fake = _fake_client(_oai_tool_resp("search", {"q": "test"}))
    oai = OpenAIClient(client=fake)
    resp = oai.complete(system=[], messages=[], model="gpt-4o")
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0]["name"] == "search"
    assert resp.tool_calls[0]["input"] == {"q": "test"}


def test_openai_client_complete_passes_model_to_sdk() -> None:
    fake = _fake_client(_oai_text_resp())
    oai = OpenAIClient(client=fake)
    oai.complete(system=[], messages=[], model="gpt-4-turbo")
    assert fake._recorded[0]["model"] == "gpt-4-turbo"


def test_openai_client_complete_passes_max_tokens() -> None:
    fake = _fake_client(_oai_text_resp())
    oai = OpenAIClient(client=fake)
    oai.complete(system=[], messages=[], model="gpt-4o", max_tokens=512)
    assert fake._recorded[0]["max_tokens"] == 512


def test_openai_client_complete_passes_temperature_when_set() -> None:
    fake = _fake_client(_oai_text_resp())
    oai = OpenAIClient(client=fake)
    oai.complete(system=[], messages=[], model="gpt-4o", temperature=0.7)
    assert fake._recorded[0]["temperature"] == 0.7


def test_openai_client_complete_omits_temperature_when_none() -> None:
    fake = _fake_client(_oai_text_resp())
    oai = OpenAIClient(client=fake)
    oai.complete(system=[], messages=[], model="gpt-4o")
    assert "temperature" not in fake._recorded[0]


def test_openai_client_complete_passes_tools_to_sdk() -> None:
    fake = _fake_client(_oai_text_resp())
    oai = OpenAIClient(client=fake)
    tools = [ToolSpec(name="fn", description="d", input_schema={"type": "object"})]
    oai.complete(system=[], messages=[], model="gpt-4o", tools=tools)
    assert "tools" in fake._recorded[0]
    assert fake._recorded[0]["tools"][0]["type"] == "function"


def test_openai_client_complete_omits_tools_when_none() -> None:
    fake = _fake_client(_oai_text_resp())
    oai = OpenAIClient(client=fake)
    oai.complete(system=[], messages=[], model="gpt-4o", tools=None)
    assert "tools" not in fake._recorded[0]


def test_openai_client_complete_omits_tools_when_empty_list() -> None:
    """An empty tools list must NOT be sent — OpenAI rejects tools=[]. """
    fake = _fake_client(_oai_text_resp())
    oai = OpenAIClient(client=fake)
    oai.complete(system=[], messages=[], model="gpt-4o", tools=[])
    assert "tools" not in fake._recorded[0]


def test_openai_client_complete_latency_seconds_positive() -> None:
    fake = _fake_client(_oai_text_resp())
    oai = OpenAIClient(client=fake)
    resp = oai.complete(system=[], messages=[], model="gpt-4o")
    assert resp.latency_seconds >= 0.0


def test_openai_client_no_cache_control_no_degraded() -> None:
    """If no cache_control in payload, degraded should be empty."""
    fake = _fake_client(_oai_text_resp())
    oai = OpenAIClient(client=fake)
    resp = oai.complete(
        system=[{"type": "text", "text": "Plain system."}],
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-4o",
    )
    assert resp.degraded == []


def test_openai_client_cache_control_in_system_sets_degraded() -> None:
    """cache_control in system block → degraded=['prompt_caching']."""
    fake = _fake_client(_oai_text_resp())
    oai = OpenAIClient(client=fake)
    resp = oai.complete(
        system=[
            {"type": "text", "text": "Sys.", "cache_control": {"type": "ephemeral"}}
        ],
        messages=[],
        model="gpt-4o",
    )
    assert "prompt_caching" in resp.degraded


def test_openai_client_cache_control_in_message_sets_degraded() -> None:
    """cache_control buried in a message block → degraded=['prompt_caching']."""
    fake = _fake_client(_oai_text_resp())
    oai = OpenAIClient(client=fake)
    resp = oai.complete(
        system=[],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "hi",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
        model="gpt-4o",
    )
    assert "prompt_caching" in resp.degraded


def test_openai_client_injected_client_not_calling_real_sdk() -> None:
    """The injected fake client is stored directly — no openai import should happen."""
    import sys

    before = set(sys.modules.keys())
    fake = _fake_client(_oai_text_resp())
    OpenAIClient(client=fake)
    after = set(sys.modules.keys())
    new_modules = after - before
    assert "openai" not in new_modules
