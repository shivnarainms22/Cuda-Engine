from pathlib import Path

from cuda_engine.config import SynthesisConfig
from cuda_engine.orchestrator import Orchestrator
from cuda_engine.services.gpu.base import CompileResult, RunResult
from cuda_engine.services.gpu.mocks import MockGPURunner
from cuda_engine.services.llm.base import LLMResponse
from cuda_engine.services.llm.mocks import MockLLMClient
from cuda_engine.services.store.mocks import InMemoryStore

SPEC_JSON = """{"name":"identity","target_arch":"sm_80","inputs":[{"name":"x","dtype":"fp32","shape":["N"]}],"outputs":[{"name":"out","dtype":"fp32","shape":["N"]}],"precision_tolerance":{"rtol":0.001,"atol":0.001},"optimization_priority":"balanced"}"""


def test_orchestrator_happy_path_with_mocks() -> None:
    torch = __import__("torch")
    store = InMemoryStore()
    orchestrator = Orchestrator(
        llm=MockLLMClient(
            responses=[
                SPEC_JSON,
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
            compile_results=[CompileResult(ok=True, so_path=Path("kernel.so"), log="ok")],
            run_results=[RunResult(ok=True, output_tensors=[torch.arange(128.0)])],
        ),
        store=store,
        cfg=SynthesisConfig(),
    )

    result = orchestrator.run(prompt="noop", reference=lambda x: x, target="sm_80")

    assert result.passed is True
    assert result.run_id
    assert result.report.spec_name == "identity"
    assert store._files[(result.run_id, "inputs/prompt.txt")] == b"noop"


def test_orchestrator_hard_gate_fails_on_correctness_mismatch() -> None:
    torch = __import__("torch")
    orchestrator = Orchestrator(
        llm=MockLLMClient(
            responses=[
                SPEC_JSON,
                LLMResponse(
                    text="```cuda\nextern code\n```",
                    model="mock",
                    tool_calls=[
                        {
                            "name": "compile_kernel",
                            "input": {"src": "extern code", "target_arch": "sm_80"},
                        }
                    ],
                ),
            ]
        ),
        gpu=MockGPURunner(
            compile_results=[CompileResult(ok=True, so_path=Path("kernel.so"), log="ok")],
            run_results=[RunResult(ok=True, output_tensors=[torch.zeros(128)])],
        ),
        store=InMemoryStore(),
        cfg=SynthesisConfig(),
    )

    result = orchestrator.run(prompt="noop", reference=lambda x: x, target="sm_80")

    assert result.passed is False
    assert result.failed_stage == 3
    assert result.correctness is not None
    assert result.correctness.passed is False
