from pathlib import Path

import pytest

from cuda_engine.models import KernelSpec, OptimizationPriority, PrecisionTolerance, TensorArg
from cuda_engine.services.gpu.base import CompileResult
from cuda_engine.services.gpu.mocks import MockGPURunner
from cuda_engine.services.llm.base import LLMResponse
from cuda_engine.services.llm.mocks import MockLLMClient
from cuda_engine.services.store.mocks import InMemoryStore
from cuda_engine.stages.base import BudgetExhaustedError
from cuda_engine.stages.codegen import Stage2Codegen


def _spec() -> KernelSpec:
    return KernelSpec(
        name="vector_add",
        target_arch="sm_80",
        inputs=[
            TensorArg(name="x", dtype="fp32", shape=("N",)),
            TensorArg(name="y", dtype="fp32", shape=("N",)),
        ],
        outputs=[TensorArg(name="out", dtype="fp32", shape=("N",))],
        precision_tolerance=PrecisionTolerance(rtol=1e-5, atol=1e-6),
        optimization_priority=OptimizationPriority.THROUGHPUT,
    )


def _compile_call(src: str = "extern code") -> LLMResponse:
    return LLMResponse(
        text=f"```cuda\n{src}\n```",
        model="mock",
        tool_calls=[
            {"name": "compile_kernel", "input": {"src": src, "target_arch": "sm_80"}},
        ],
    )


def test_stage2_codegen_happy_path_routes_compile_tool_call() -> None:
    store = InMemoryStore()
    gpu = MockGPURunner(
        compile_results=[CompileResult(ok=True, so_path=Path("/tmp/kernel.so"), log="ok")]
    )
    stage = Stage2Codegen(llm=MockLLMClient([_compile_call()]), gpu=gpu, store=store)

    artifact = stage.run(spec=_spec(), run_id="run123", retry_budget=3)

    assert artifact.kernel_so_path == Path("/tmp/kernel.so")
    assert store._files[("run123", "stage2_codegen/attempt_01/kernel.cu")] == b"extern code"
    assert store._files[("run123", "stage2_codegen/final/kernel.cu")] == b"extern code"


def test_stage2_codegen_retries_after_compile_error() -> None:
    llm = MockLLMClient([_compile_call("bad"), _compile_call("fixed")])
    gpu = MockGPURunner(
        compile_results=[
            CompileResult(ok=False, log="bad", errors=["bad"]),
            CompileResult(ok=True, so_path=Path("/tmp/kernel.so"), log="ok"),
        ]
    )
    stage = Stage2Codegen(llm=llm, gpu=gpu, store=InMemoryStore())

    artifact = stage.run(spec=_spec(), run_id="run123", retry_budget=3)

    assert artifact.kernel_cu_path.as_posix().endswith("stage2_codegen/final/kernel.cu")
    assert artifact.kernel_so_path == Path("/tmp/kernel.so")
    assert llm.call_count == 2


def test_stage2_codegen_raises_when_retry_budget_exhausted() -> None:
    stage = Stage2Codegen(
        llm=MockLLMClient([_compile_call("bad1"), _compile_call("bad2")]),
        gpu=MockGPURunner(
            compile_results=[
                CompileResult(ok=False, log="bad1", errors=["bad1"]),
                CompileResult(ok=False, log="bad2", errors=["bad2"]),
            ]
        ),
        store=InMemoryStore(),
    )

    with pytest.raises(BudgetExhaustedError, match="codegen exhausted retry budget"):
        stage.run(spec=_spec(), run_id="run123", retry_budget=2)
