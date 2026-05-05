from cuda_engine.services.llm.base import LLMResponse
from cuda_engine.services.llm.mocks import MockLLMClient


def test_mock_llm_returns_canned_response() -> None:
    mock = MockLLMClient(responses=["hello", "world"])

    first = mock.complete(system=[], messages=[], tools=None, model="claude-sonnet-4-6")
    second = mock.complete(system=[], messages=[], tools=None, model="claude-sonnet-4-6")

    assert first.text == "hello"
    assert second.text == "world"
    assert mock.call_count == 2


def test_mock_llm_supports_tool_calls() -> None:
    canned = LLMResponse(
        text="",
        tool_calls=[{"name": "compile_kernel", "input": {"src": "..."}}],
        model="mock",
        tokens_in=10,
        tokens_out=5,
    )
    mock = MockLLMClient(responses=[canned])

    response = mock.complete(system=[], messages=[], tools=None, model="mock")

    assert response.tool_calls[0]["name"] == "compile_kernel"
