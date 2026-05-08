from pathlib import Path

import pytest

from cuda_engine.models import (
    CorrectnessReport,
    KernelSpec,
    OptimizationPriority,
    PrecisionTolerance,
    TensorArg,
)
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

    artifact = stage.run(spec=_spec(), run_id="run123", retry_budget=3, model="claude-sonnet-4-6")

    assert artifact.kernel_so_path == Path("/tmp/kernel.so")
    assert store._files[("run123", "stage2_codegen/attempt_01/kernel.cu")] == b"extern code"
    assert store._files[("run123", "stage2_codegen/final/kernel.cu")] == b"extern code"


def test_stage2_codegen_retries_after_compile_error() -> None:
    store = InMemoryStore()
    llm = MockLLMClient([_compile_call("bad"), _compile_call("fixed")])
    gpu = MockGPURunner(
        compile_results=[
            CompileResult(ok=False, log="bad", errors=["bad"]),
            CompileResult(ok=True, so_path=Path("/tmp/kernel.so"), log="ok"),
        ]
    )
    stage = Stage2Codegen(llm=llm, gpu=gpu, store=store)

    artifact = stage.run(spec=_spec(), run_id="run123", retry_budget=3, model="claude-sonnet-4-6")

    assert artifact.kernel_cu_path.as_posix().endswith("stage2_codegen/final/kernel.cu")
    assert artifact.kernel_so_path == Path("/tmp/kernel.so")
    assert llm.call_count == 2
    assert store._files[("run123", "stage2_codegen/attempt_01/compile_log.txt")] == b"bad"
    assert "Compilation failed" in llm.calls[1]["messages"][-1]["content"]
    assert "Compile log:\nbad" in llm.calls[1]["messages"][-1]["content"]


def test_stage2_codegen_accepts_correctness_repair_context_and_custom_prefix() -> None:
    store = InMemoryStore()
    llm = MockLLMClient([_compile_call("fixed")])
    stage = Stage2Codegen(
        llm=llm,
        gpu=MockGPURunner(
            compile_results=[CompileResult(ok=True, so_path=Path("/tmp/kernel.so"), log="ok")]
        ),
        store=store,
    )
    correctness = CorrectnessReport(
        passed=False,
        max_abs_err=1.0,
        max_rel_err=1.0,
        shapes_tested=[(128,)],
        failing_inputs=[{"shape": (128,), "max_abs_err": 1.0}],
    )

    artifact = stage.run(
        spec=_spec(),
        run_id="run123",
        retry_budget=1,
        model="claude-sonnet-4-6",
        repair_context=correctness,
        artifact_prefix="stage3_repair/attempt_01/codegen",
    )

    prompt = llm.calls[0]["messages"][0]["content"]
    assert "Repair kernel.cu" in prompt
    assert "failing_inputs" in prompt
    assert artifact.kernel_cu_path.as_posix().endswith("stage3_repair/attempt_01/codegen/final/kernel.cu")
    assert store._files[("run123", "stage3_repair/attempt_01/codegen/attempt_01/kernel.cu")] == b"fixed"


def test_stage2_codegen_raises_when_retry_budget_exhausted() -> None:
    store = InMemoryStore()
    stage = Stage2Codegen(
        llm=MockLLMClient([_compile_call("bad1"), _compile_call("bad2")]),
        gpu=MockGPURunner(
            compile_results=[
                CompileResult(ok=False, log="bad1", errors=["bad1"]),
                CompileResult(ok=False, log="last nvcc log", errors=["bad2"]),
            ]
        ),
        store=store,
    )

    with pytest.raises(BudgetExhaustedError) as exc_info:
        stage.run(spec=_spec(), run_id="run123", retry_budget=2, model="claude-sonnet-4-6")
    assert "codegen exhausted retry budget" in str(exc_info.value)
    assert "last nvcc log" in str(exc_info.value)
    assert store._files[("run123", "stage2_codegen/attempt_01/compile_log.txt")] == b"bad1"
    assert store._files[("run123", "stage2_codegen/attempt_02/compile_log.txt")] == b"last nvcc log"


def test_stage2_codegen_budget_exhausted_carries_summary() -> None:
    store = InMemoryStore()
    stage = Stage2Codegen(
        llm=MockLLMClient([_compile_call("bad1"), _compile_call("bad2")]),
        gpu=MockGPURunner(
            compile_results=[
                CompileResult(ok=False, log="log-1", errors=["err-1"]),
                CompileResult(ok=False, log="log-2", errors=["err-2"]),
            ]
        ),
        store=store,
    )

    with pytest.raises(BudgetExhaustedError) as exc_info:
        stage.run(spec=_spec(), run_id="run123", retry_budget=2, model="claude-sonnet-4-6")

    summary = exc_info.value.summary
    assert summary is not None
    assert summary.attempts_made == 2
    assert "err-2" in summary.last_compile_errors
    assert "log-2" in summary.last_compile_log
    assert summary.last_source_attempt == "bad2"
