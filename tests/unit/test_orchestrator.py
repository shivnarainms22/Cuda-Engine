from cuda_engine.config import SynthesisConfig
from cuda_engine.orchestrator import Orchestrator
from cuda_engine.services.gpu.mocks import MockGPURunner
from cuda_engine.services.llm.mocks import MockLLMClient
from cuda_engine.services.store.mocks import InMemoryStore


def test_orchestrator_happy_path_with_mocks() -> None:
    orchestrator = Orchestrator(
        llm=MockLLMClient(responses=[]),
        gpu=MockGPURunner(),
        store=InMemoryStore(),
        cfg=SynthesisConfig(),
    )

    result = orchestrator.run(prompt="noop", reference=lambda x: x, target="sm_80")

    assert result.passed is True
    assert result.run_id
    assert result.report.spec_name == "placeholder"
