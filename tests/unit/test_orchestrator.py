from cuda_engine.config import SynthesisConfig
from cuda_engine.orchestrator import Orchestrator
from cuda_engine.services.gpu.base import CompileResult
from cuda_engine.services.gpu.mocks import MockGPURunner
from cuda_engine.services.llm.base import LLMResponse
from cuda_engine.services.llm.mocks import MockLLMClient
from cuda_engine.services.store.mocks import InMemoryStore


def test_orchestrator_happy_path_with_mocks() -> None:
    store = InMemoryStore()
    orchestrator = Orchestrator(
        llm=MockLLMClient(
            responses=[
                LLMResponse(
                    text="```cuda\nextern code\n```",
                    model="mock",
                    tool_calls=[
                        {
                            "name": "compile_kernel",
                            "input": {"src": "extern code", "target_arch": "sm_80"},
                        }
                    ],
                )
            ]
        ),
        gpu=MockGPURunner(
            compile_results=[CompileResult(ok=True, log="ok")],
        ),
        store=store,
        cfg=SynthesisConfig(),
    )

    result = orchestrator.run(prompt="noop", reference=lambda x: x, target="sm_80")

    assert result.passed is True
    assert result.run_id
    assert result.report.spec_name == "placeholder"
    assert store._files[(result.run_id, "inputs/prompt.txt")] == b"noop"
