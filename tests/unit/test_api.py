from cuda_engine import SynthesisConfig, SynthesisResult, synthesize
from cuda_engine.services.gpu.mocks import MockGPURunner
from cuda_engine.services.llm.mocks import MockLLMClient
from cuda_engine.services.store.mocks import InMemoryStore


def test_synthesize_returns_result_with_mocks() -> None:
    result = synthesize(
        prompt="noop",
        reference=lambda x: x,
        target="sm_80",
        config=SynthesisConfig(),
        _llm=MockLLMClient(responses=[]),
        _gpu=MockGPURunner(),
        _store=InMemoryStore(),
    )

    assert isinstance(result, SynthesisResult)
    assert result.passed
