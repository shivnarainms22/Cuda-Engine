from types import SimpleNamespace

from cuda_engine.config import SynthesisConfig
from cuda_engine.services.llm.anthropic import AnthropicClient
from cuda_engine.services.llm.tools import COMPILE_KERNEL


class _FakeMessages:
    def __init__(self) -> None:
        self.kwargs = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            model="claude-sonnet-4-6",
            content=[
                SimpleNamespace(type="text", text="generated"),
                SimpleNamespace(type="tool_use", name="compile_kernel", input={"src": "code"}),
            ],
            usage=SimpleNamespace(
                input_tokens=11,
                output_tokens=7,
                cache_read_input_tokens=5,
            ),
        )


class _FakeAnthropic:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self.messages = _FakeMessages()


def test_anthropic_client_translates_request_and_parses_response(monkeypatch) -> None:
    fake_holder = {}

    def fake_factory(api_key: str | None = None):
        fake = _FakeAnthropic(api_key=api_key)
        fake_holder["client"] = fake
        return fake

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("anthropic.Anthropic", fake_factory)
    client = AnthropicClient(cfg=SynthesisConfig())

    response = client.complete(
        system=[{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": "hello"}],
        tools=[COMPILE_KERNEL],
        model="claude-sonnet-4-6",
        max_tokens=128,
        temperature=0.2,
    )

    fake = fake_holder["client"]
    assert fake.api_key == "test-key"
    assert fake.messages.kwargs["tools"][0]["name"] == "compile_kernel"
    assert fake.messages.kwargs["system"][0]["cache_control"]["type"] == "ephemeral"
    assert response.text == "generated"
    assert response.tool_calls == [{"name": "compile_kernel", "input": {"src": "code"}}]
    assert response.tokens_in == 11
    assert response.tokens_out == 7
    assert response.cache_read_tokens == 5
    assert response.latency_seconds >= 0.0


def test_anthropic_client_omits_tools_when_none(monkeypatch) -> None:
    fake_holder = {}

    def fake_factory(api_key: str | None = None):
        fake = _FakeAnthropic(api_key=api_key)
        fake_holder["client"] = fake
        return fake

    monkeypatch.setattr("anthropic.Anthropic", fake_factory)
    client = AnthropicClient(cfg=SynthesisConfig())

    client.complete(
        system=[{"type": "text", "text": "sys"}],
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        model="claude-opus-4-7",
    )

    fake = fake_holder["client"]
    assert "tools" not in fake.messages.kwargs


def test_anthropic_client_omits_temperature_by_default(monkeypatch) -> None:
    fake_holder = {}

    def fake_factory(api_key: str | None = None):
        fake = _FakeAnthropic(api_key=api_key)
        fake_holder["client"] = fake
        return fake

    monkeypatch.setattr("anthropic.Anthropic", fake_factory)
    client = AnthropicClient(cfg=SynthesisConfig())

    client.complete(
        system=[{"type": "text", "text": "sys"}],
        messages=[{"role": "user", "content": "hello"}],
        model="claude-opus-4-7",
    )

    fake = fake_holder["client"]
    assert "temperature" not in fake.messages.kwargs
