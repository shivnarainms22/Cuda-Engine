"""Tests for GeminiClient — no real google-genai SDK needed; client is injected."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from cuda_engine.services.llm.base import LLMResponse, ToolSpec
from cuda_engine.services.llm.gemini_client import GeminiClient

# ---------------------------------------------------------------------------
# Fake Gemini client helpers
# ---------------------------------------------------------------------------


def _fake_client(resp: Any) -> SimpleNamespace:
    """Fake that mimics google.genai.Client().models.generate_content(...)."""
    recorded: list[dict] = []

    def generate_content(**kwargs: Any) -> Any:
        recorded.append(kwargs)
        return resp

    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content),
        _recorded=recorded,
    )
    return client


def _gemini_text_resp(text: str = "hello", tokens_in: int = 10, tokens_out: int = 5) -> Any:
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text=text, function_call=None)]
                )
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=tokens_in,
            candidates_token_count=tokens_out,
        ),
    )


def _gemini_tool_resp(name: str = "lookup", args: dict | None = None) -> Any:
    args = args or {"key": "val"}
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            text=None,
                            function_call=SimpleNamespace(name=name, args=args),
                        )
                    ]
                )
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=8,
            candidates_token_count=3,
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_gemini_client_capabilities_provider() -> None:
    gc = GeminiClient(client=_fake_client(_gemini_text_resp()))
    assert gc.capabilities.provider == "gemini"


def test_gemini_client_capabilities_supports_tool_use_and_caching() -> None:
    gc = GeminiClient(client=_fake_client(_gemini_text_resp()))
    caps = gc.capabilities
    assert caps.tool_use is True
    assert caps.prompt_caching is True


def test_gemini_client_complete_text_response() -> None:
    fake = _fake_client(_gemini_text_resp("Gemini response", 12, 6))
    gc = GeminiClient(client=fake)
    resp = gc.complete(
        system=[], messages=[{"role": "user", "content": "hello"}], model="gemini-2.0-flash"
    )
    assert isinstance(resp, LLMResponse)
    assert resp.text == "Gemini response"
    assert resp.provider == "gemini"
    assert resp.model == "gemini-2.0-flash"
    assert resp.tokens_in == 12
    assert resp.tokens_out == 6


def test_gemini_client_complete_tool_calls_mapped() -> None:
    fake = _fake_client(_gemini_tool_resp("search", {"q": "cats"}))
    gc = GeminiClient(client=fake)
    resp = gc.complete(system=[], messages=[], model="gemini-2.0-flash")
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0]["name"] == "search"
    assert resp.tool_calls[0]["input"] == {"q": "cats"}


def test_gemini_client_complete_passes_model_to_sdk() -> None:
    fake = _fake_client(_gemini_text_resp())
    gc = GeminiClient(client=fake)
    gc.complete(system=[], messages=[], model="gemini-1.5-pro")
    assert fake._recorded[0]["model"] == "gemini-1.5-pro"


def test_gemini_client_complete_passes_contents_to_sdk() -> None:
    fake = _fake_client(_gemini_text_resp())
    gc = GeminiClient(client=fake)
    gc.complete(
        system=[],
        messages=[{"role": "user", "content": "ping"}],
        model="gemini-2.0-flash",
    )
    contents = fake._recorded[0]["contents"]
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == "ping"


def test_gemini_client_complete_passes_config_to_sdk() -> None:
    fake = _fake_client(_gemini_text_resp())
    gc = GeminiClient(client=fake)
    gc.complete(system=[], messages=[], model="gemini-2.0-flash", max_tokens=256)
    config = fake._recorded[0]["config"]
    assert config["max_output_tokens"] == 256


def test_gemini_client_complete_passes_temperature_when_set() -> None:
    fake = _fake_client(_gemini_text_resp())
    gc = GeminiClient(client=fake)
    gc.complete(system=[], messages=[], model="gemini-2.0-flash", temperature=0.3)
    assert fake._recorded[0]["config"]["temperature"] == 0.3


def test_gemini_client_complete_omits_temperature_when_none() -> None:
    fake = _fake_client(_gemini_text_resp())
    gc = GeminiClient(client=fake)
    gc.complete(system=[], messages=[], model="gemini-2.0-flash")
    assert "temperature" not in fake._recorded[0]["config"]


def test_gemini_client_complete_includes_system_instruction_in_config() -> None:
    fake = _fake_client(_gemini_text_resp())
    gc = GeminiClient(client=fake)
    gc.complete(
        system=[{"type": "text", "text": "Be helpful."}],
        messages=[],
        model="gemini-2.0-flash",
    )
    assert fake._recorded[0]["config"]["system_instruction"] == "Be helpful."


def test_gemini_client_complete_tools_included_in_config() -> None:
    fake = _fake_client(_gemini_text_resp())
    gc = GeminiClient(client=fake)
    tools = [ToolSpec(name="fn", description="d", input_schema={"type": "object"})]
    gc.complete(system=[], messages=[], model="gemini-2.0-flash", tools=tools)
    config_tools = fake._recorded[0]["config"]["tools"]
    assert len(config_tools) == 1
    assert config_tools[0]["function_declarations"][0]["name"] == "fn"


def test_gemini_client_complete_omits_tools_in_config_when_none() -> None:
    fake = _fake_client(_gemini_text_resp())
    gc = GeminiClient(client=fake)
    gc.complete(system=[], messages=[], model="gemini-2.0-flash", tools=None)
    assert "tools" not in fake._recorded[0]["config"]


def test_gemini_client_complete_latency_seconds_positive() -> None:
    fake = _fake_client(_gemini_text_resp())
    gc = GeminiClient(client=fake)
    resp = gc.complete(system=[], messages=[], model="gemini-2.0-flash")
    assert resp.latency_seconds >= 0.0


def test_gemini_client_injected_client_does_not_import_google_genai() -> None:
    """The injected fake should not trigger a real SDK import."""
    import sys

    before = set(sys.modules.keys())
    fake = _fake_client(_gemini_text_resp())
    GeminiClient(client=fake)
    after = set(sys.modules.keys())
    new_modules = after - before
    assert "google" not in new_modules
    assert "genai" not in new_modules
