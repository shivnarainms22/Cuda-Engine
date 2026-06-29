from cuda_engine.services.llm.base import LLMResponse
from cuda_engine.services.llm.mocks import MockLLMClient


def test_llm_response_provider_defaults_to_empty_string() -> None:
    r = LLMResponse(text="hi", model="m")
    assert r.provider == ""


def test_llm_response_degraded_defaults_to_empty_list() -> None:
    r = LLMResponse(text="hi", model="m")
    assert r.degraded == []


def test_llm_response_degraded_is_independent_across_instances() -> None:
    a = LLMResponse(text="a", model="m")
    b = LLMResponse(text="b", model="m")
    assert a.degraded is not b.degraded


def test_mock_client_exposes_capabilities() -> None:
    mock = MockLLMClient(responses=[])
    caps = mock.capabilities
    assert caps.provider == "mock"
    assert caps.prompt_caching is False
    assert caps.tool_use is False


def test_mock_client_sets_provider_on_response() -> None:
    mock = MockLLMClient(responses=["hello"])
    response = mock.complete(system=[], messages=[], model="mock")
    assert response.provider == "mock"
