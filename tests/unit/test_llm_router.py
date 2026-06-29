import pytest

from cuda_engine.services.llm.mocks import MockLLMClient
from cuda_engine.services.llm.router import LLMRouter, parse_model_id


def test_parse_model_id_with_provider_prefix() -> None:
    assert parse_model_id("openai:gpt-4") == ("openai", "gpt-4")


def test_parse_model_id_bare_id_defaults_to_anthropic() -> None:
    assert parse_model_id("claude-sonnet-4-6") == ("anthropic", "claude-sonnet-4-6")


def test_parse_model_id_preserves_colon_in_bare_model() -> None:
    # only splits on first colon
    assert parse_model_id("openai:gpt-4:latest") == ("openai", "gpt-4:latest")


def test_router_dispatches_to_correct_provider() -> None:
    anthropic_mock = MockLLMClient(responses=["from-anthropic"])
    openai_mock = MockLLMClient(responses=["from-openai"])
    router = LLMRouter(providers={"anthropic": anthropic_mock, "openai": openai_mock})

    response = router.complete(system=[], messages=[], model="openai:gpt-4")
    assert response.text == "from-openai"
    assert openai_mock.calls[0]["model"] == "gpt-4"
    assert anthropic_mock.call_count == 0


def test_router_strips_prefix_from_model_string() -> None:
    mock = MockLLMClient(responses=["ok"])
    router = LLMRouter(providers={"anthropic": mock})

    router.complete(system=[], messages=[], model="anthropic:claude-opus-4-7")
    assert mock.calls[0]["model"] == "claude-opus-4-7"


def test_router_bare_model_routes_to_anthropic() -> None:
    mock = MockLLMClient(responses=["ok"])
    router = LLMRouter(providers={"anthropic": mock})

    router.complete(system=[], messages=[], model="claude-sonnet-4-6")
    assert mock.calls[0]["model"] == "claude-sonnet-4-6"


def test_router_unknown_provider_raises_key_error() -> None:
    mock = MockLLMClient(responses=[])
    router = LLMRouter(providers={"anthropic": mock})

    with pytest.raises(KeyError):
        router.complete(system=[], messages=[], model="openai:gpt-4")


def test_router_capabilities_returns_anthropic_provider_caps() -> None:
    anthropic_mock = MockLLMClient(responses=[])
    openai_mock = MockLLMClient(responses=[])
    router = LLMRouter(providers={"anthropic": anthropic_mock, "openai": openai_mock})

    caps = router.capabilities
    assert caps.provider == "mock"  # MockLLMClient.capabilities.provider


def test_router_capabilities_falls_back_to_first_if_no_anthropic() -> None:
    openai_mock = MockLLMClient(responses=[])
    router = LLMRouter(providers={"openai": openai_mock})

    caps = router.capabilities
    assert caps.provider == "mock"


def test_router_passes_kwargs_to_provider() -> None:
    mock = MockLLMClient(responses=["ok"])
    router = LLMRouter(providers={"anthropic": mock})

    router.complete(
        system=[{"type": "text", "text": "sys"}],
        messages=[{"role": "user", "content": "hi"}],
        model="anthropic:claude-sonnet-4-6",
        max_tokens=512,
        temperature=0.5,
    )
    call = mock.calls[0]
    assert call["max_tokens"] == 512
    assert call["temperature"] == 0.5
    assert call["system"] == [{"type": "text", "text": "sys"}]
