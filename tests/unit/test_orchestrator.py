import json
from pathlib import Path

import pytest

from cuda_engine.config import RetryBudgets, SynthesisConfig
from cuda_engine.orchestrator import Orchestrator
from cuda_engine.services.gpu.base import BenchmarkResult, CompileResult, NsightMetrics, RunResult
from cuda_engine.services.gpu.mocks import MockGPURunner
from cuda_engine.services.llm.base import LLMResponse
from cuda_engine.services.llm.mocks import MockLLMClient
from cuda_engine.services.store.mocks import InMemoryStore
from cuda_engine.stages.base import BudgetExhaustedError

SPEC_JSON = """{"name":"identity","target_arch":"sm_80","inputs":[{"name":"x","dtype":"fp32","shape":["N"]}],"outputs":[{"name":"out","dtype":"fp32","shape":["N"]}],"precision_tolerance":{"rtol":0.001,"atol":0.001},"optimization_priority":"balanced"}"""
MATRIX_SPEC_JSON = """{"name":"matrix_identity","target_arch":"sm_80","inputs":[{"name":"x","dtype":"fp32","shape":["B","D"]}],"outputs":[{"name":"out","dtype":"fp32","shape":["B","D"]}],"precision_tolerance":{"rtol":0.001,"atol":0.001},"optimization_priority":"balanced"}"""

SHAPE_SIZES = (0, 1, 127, 128, 1024, 4097)


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
                ),
                "```cuda\n// annotated\nextern code\n```",
            ]
        ),
        gpu=MockGPURunner(
            compile_results=[CompileResult(ok=True, so_path=Path("kernel.so"), log="ok")],
            run_results=[
                RunResult(ok=True, output_tensors=[torch.arange(size, dtype=torch.float32)])
                for size in SHAPE_SIZES
            ],
        ),
        store=store,
        cfg=SynthesisConfig(
            performance_shape_n=256,
            benchmark_warmup_iterations=2,
            benchmark_timed_iterations=3,
        ),
    )

    result = orchestrator.run(prompt="noop", reference=lambda x: x, target="sm_80")

    assert result.passed is True
    assert result.run_id
    assert result.report.spec_name == "identity"
    assert [trace.stage_name for trace in result.report.stage_traces] == [
        "interview",
        "codegen",
        "correctness",
        "performance",
        "polish",
    ]
    assert result.report.total_llm_tokens_in > 0
    assert result.report.total_llm_tokens_out > 0
    assert all(trace.succeeded for trace in result.report.stage_traces)
    assert store._files[(result.run_id, "inputs/prompt.txt")] == b"noop"
    persisted = json.loads(store._files[(result.run_id, "report.json")])
    assert persisted["passed"] is True
    assert persisted["failed_stage"] is None
    assert persisted["report"]["stages_executed"] == [
        "interview",
        "codegen",
        "correctness",
        "performance",
        "polish",
    ]
    assert persisted["correctness"]["passed"] is True
    assert persisted["performance"]["below_target"] is False
    assert orchestrator.gpu.benchmark_calls[0]["input_shapes"] == [(256,)]
    assert orchestrator.gpu.benchmark_calls[0]["warmup_iterations"] == 2
    assert orchestrator.gpu.benchmark_calls[0]["timed_iterations"] == 3


def test_orchestrator_passes_configured_correctness_shapes() -> None:
    torch = __import__("torch")
    store = InMemoryStore()
    orchestrator = Orchestrator(
        llm=MockLLMClient(
            responses=[
                MATRIX_SPEC_JSON,
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
                "```cuda\n// annotated\nextern code\n```",
            ]
        ),
        gpu=MockGPURunner(
            compile_results=[CompileResult(ok=True, so_path=Path("kernel.so"), log="ok")],
            run_results=[
                RunResult(ok=True, output_tensors=[torch.arange(6, dtype=torch.float32).reshape(2, 3)]),
                RunResult(ok=True, output_tensors=[torch.arange(20, dtype=torch.float32).reshape(4, 5)]),
            ],
        ),
        store=store,
        cfg=SynthesisConfig(correctness_shapes=((2, 3), (4, 5))),
    )

    result = orchestrator.run(prompt="noop", reference=lambda x: x, target="sm_80")

    assert result.passed is True
    assert result.correctness is not None
    assert result.correctness.shapes_tested == [(2, 3), (4, 5)]


def test_orchestrator_hard_gate_fails_on_correctness_mismatch() -> None:
    torch = __import__("torch")
    store = InMemoryStore()
    run_results = [
        RunResult(ok=True, output_tensors=[torch.arange(size, dtype=torch.float32)])
        for size in (0, 1, 127)
    ]
    run_results.append(RunResult(ok=True, output_tensors=[torch.zeros(128)]))
    run_results.extend(
        RunResult(ok=True, output_tensors=[torch.arange(size, dtype=torch.float32)])
        for size in (1024, 4097)
    )
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
                "```cuda\n// unused because correctness fails\nextern code\n```",
            ]
        ),
        gpu=MockGPURunner(
            compile_results=[CompileResult(ok=True, so_path=Path("kernel.so"), log="ok")],
            run_results=run_results,
        ),
        store=store,
        cfg=SynthesisConfig(retry_budgets=RetryBudgets(correctness=0)),
    )

    result = orchestrator.run(prompt="noop", reference=lambda x: x, target="sm_80")

    assert result.passed is False
    assert result.failed_stage == 3
    assert result.correctness is not None
    assert result.correctness.passed is False
    assert [trace.stage_name for trace in result.report.stage_traces] == [
        "interview",
        "codegen",
        "correctness",
    ]
    assert result.report.stage_traces[-1].succeeded is False
    persisted = json.loads(store._files[(result.run_id, "report.json")])
    assert persisted["passed"] is False
    assert persisted["failed_stage"] == 3
    assert persisted["failure_reason"] == "correctness check failed"
    assert persisted["report"]["stage_traces"][-1]["succeeded"] is False
    assert persisted["correctness"]["passed"] is False
    assert persisted["performance"] is None


def test_orchestrator_repairs_after_correctness_failure() -> None:
    torch = __import__("torch")
    store = InMemoryStore()
    llm = MockLLMClient(
        responses=[
            SPEC_JSON,
            LLMResponse(
                text="```cuda\nbad code\n```",
                model="mock",
                tool_calls=[
                    {
                        "name": "compile_kernel",
                        "input": {"src": "bad code", "target_arch": "sm_80"},
                    }
                ],
            ),
            LLMResponse(
                text="```cuda\nfixed code\n```",
                model="mock",
                tool_calls=[
                    {
                        "name": "compile_kernel",
                        "input": {"src": "fixed code", "target_arch": "sm_80"},
                    }
                ],
            ),
            "```cuda\n// annotated\nfixed code\n```",
        ]
    )
    first_correctness = [
        RunResult(ok=True, output_tensors=[torch.arange(size, dtype=torch.float32)])
        for size in (0, 1, 127)
    ]
    first_correctness.append(RunResult(ok=True, output_tensors=[torch.zeros(128)]))
    first_correctness.extend(
        RunResult(ok=True, output_tensors=[torch.arange(size, dtype=torch.float32)])
        for size in (1024, 4097)
    )
    repaired_correctness = [
        RunResult(ok=True, output_tensors=[torch.arange(size, dtype=torch.float32)])
        for size in SHAPE_SIZES
    ]
    orchestrator = Orchestrator(
        llm=llm,
        gpu=MockGPURunner(
            compile_results=[
                CompileResult(ok=True, so_path=Path("bad.so"), log="ok"),
                CompileResult(ok=True, so_path=Path("fixed.so"), log="ok"),
            ],
            run_results=[*first_correctness, *repaired_correctness],
        ),
        store=store,
        cfg=SynthesisConfig(),
    )

    result = orchestrator.run(prompt="noop", reference=lambda x: x, target="sm_80")

    assert result.passed is True
    assert result.correctness is not None
    assert result.correctness.passed is True
    assert llm.call_count == 4
    assert b'"passed": false' in store._files[
        (result.run_id, "stage3_repair/attempt_01/correctness_report.json")
    ]
    repair_prompt = llm.calls[2]["messages"][0]["content"]
    assert "Repair kernel.cu" in repair_prompt
    assert "failing_inputs" in repair_prompt
    assert store._files[
        (result.run_id, "stage3_repair/attempt_01/codegen/attempt_01/kernel.cu")
    ] == b"fixed code"


def test_orchestrator_escalates_codegen_to_opus_on_bust() -> None:
    """Sonnet busts 3x on codegen, Opus succeeds 1st try → run completes via Opus."""
    torch = __import__("torch")
    store = InMemoryStore()

    def _fail_compile_response() -> LLMResponse:
        return LLMResponse(
            text="```cuda\nbroken\n```",
            model="claude-sonnet-4-6",
            tool_calls=[
                {"name": "compile_kernel", "input": {"src": "broken", "target_arch": "sm_80"}}
            ],
        )

    def _ok_compile_response() -> LLMResponse:
        return LLMResponse(
            text="```cuda\ngood\n```",
            model="claude-opus-4-7",
            tool_calls=[
                {"name": "compile_kernel", "input": {"src": "good", "target_arch": "sm_80"}}
            ],
        )

    orchestrator = Orchestrator(
        llm=MockLLMClient(
            responses=[
                SPEC_JSON,                                    # interview (sonnet)
                _fail_compile_response(),                     # codegen sonnet attempt 1
                _fail_compile_response(),                     # codegen sonnet attempt 2
                _fail_compile_response(),                     # codegen sonnet attempt 3 (bust)
                _ok_compile_response(),                       # codegen opus attempt 1
                "```cuda\n// annotated\ngood\n```",           # polish
            ]
        ),
        gpu=MockGPURunner(
            compile_results=[
                CompileResult(ok=False, log="bad1", errors=["err1"]),
                CompileResult(ok=False, log="bad2", errors=["err2"]),
                CompileResult(ok=False, log="bad3", errors=["err3"]),
                CompileResult(ok=True, so_path=Path("/tmp/kernel.so"), log="ok"),
                CompileResult(ok=True, so_path=Path("/tmp/kernel.so"), log="ok"),  # polish recompile
            ],
            run_results=[
                RunResult(ok=True, output_tensors=[torch.arange(size, dtype=torch.float32)])
                for size in SHAPE_SIZES * 2  # both initial correctness + polish-correctness
            ],
        ),
        store=store,
        cfg=SynthesisConfig(
            retry_budgets=RetryBudgets(codegen=3),
            performance_shape_n=256,
            benchmark_warmup_iterations=2,
            benchmark_timed_iterations=3,
        ),
    )

    result = orchestrator.run(prompt="noop", reference=lambda x: x, target="sm_80")

    assert result.passed is True
    codegen_trace = next(t for t in result.report.stage_traces if t.stage_name == "codegen")
    assert "claude-sonnet-4-6" in codegen_trace.model_used
    assert "claude-opus-4-7" in codegen_trace.model_used
    assert codegen_trace.attempts == 4  # 3 sonnet + 1 opus
    # Opus retry artifact lands under escalated/
    escalated_files = [
        key for key in store._files if "stage2_codegen/escalated/attempt_01/" in key[1]
    ]
    assert escalated_files, f"expected escalated/ files, got: {list(store._files.keys())[:20]}"


def test_orchestrator_codegen_escalation_disabled_surfaces_bust() -> None:
    """With escalate_to_opus_on_bust=False, Sonnet bust propagates as Stage-3 failure."""
    store = InMemoryStore()

    def _fail_compile_response() -> LLMResponse:
        return LLMResponse(
            text="```cuda\nbroken\n```",
            model="mock",
            tool_calls=[
                {"name": "compile_kernel", "input": {"src": "broken", "target_arch": "sm_80"}}
            ],
        )

    orchestrator = Orchestrator(
        llm=MockLLMClient(
            responses=[
                SPEC_JSON,
                _fail_compile_response(), _fail_compile_response(), _fail_compile_response(),
            ]
        ),
        gpu=MockGPURunner(
            compile_results=[
                CompileResult(ok=False, log="bad1", errors=["err1"]),
                CompileResult(ok=False, log="bad2", errors=["err2"]),
                CompileResult(ok=False, log="bad3", errors=["err3"]),
            ],
        ),
        store=store,
        cfg=SynthesisConfig(
            escalate_to_opus_on_bust=False,
            retry_budgets=RetryBudgets(codegen=3),
        ),
    )

    with pytest.raises(BudgetExhaustedError):
        orchestrator.run(prompt="noop", reference=lambda x: x, target="sm_80")


def test_orchestrator_perf_stage_escalates_to_opus() -> None:
    """End-to-end through Orchestrator: perf below target on Sonnet → Opus iteration runs."""
    torch = __import__("torch")
    store = InMemoryStore()

    def _perf_fix_response(src: str, model: str) -> LLMResponse:
        return LLMResponse(
            text=f"```cuda\n{src}\n```",
            model=model,
            tool_calls=[
                {"name": "compile_kernel", "input": {"src": src, "target_arch": "sm_80"}}
            ],
        )

    orchestrator = Orchestrator(
        llm=MockLLMClient(
            responses=[
                SPEC_JSON,                                             # interview
                LLMResponse(                                           # codegen
                    text="```cuda\ninitial kernel\n```",
                    model="claude-sonnet-4-6",
                    tool_calls=[{"name": "compile_kernel", "input": {"src": "initial kernel", "target_arch": "sm_80"}}],
                ),
                _perf_fix_response("// sonnet perf fix", "claude-sonnet-4-6"),  # perf retry sonnet
                _perf_fix_response("// opus perf fix", "claude-opus-4-7"),      # perf retry opus
                "```cuda\n// annotated\ninitial kernel\n```",                   # polish
            ]
        ),
        gpu=MockGPURunner(
            compile_results=[
                CompileResult(ok=True, so_path=Path("/tmp/kernel.so"), log="ok"),   # codegen
                CompileResult(ok=True, so_path=Path("/tmp/v2.so"), log="ok"),       # sonnet perf retry
                CompileResult(ok=True, so_path=Path("/tmp/v3.so"), log="ok"),       # opus perf retry
                CompileResult(ok=True, so_path=Path("/tmp/kernel.so"), log="ok"),   # polish recompile
            ],
            run_results=[
                RunResult(ok=True, output_tensors=[torch.arange(size, dtype=torch.float32)])
                for size in SHAPE_SIZES
            ],
            benchmark_results=[
                # initial benchmark → below target (custom_ms > baseline_ms)
                BenchmarkResult(ok=True, custom_ms=4.0, baseline_ms=2.0, warmup_iterations=2, timed_iterations=3),
                # sonnet perf retry after-recompile → still below target
                BenchmarkResult(ok=True, custom_ms=3.0, baseline_ms=2.0, warmup_iterations=2, timed_iterations=3),
                # opus perf retry after-recompile → above target
                BenchmarkResult(ok=True, custom_ms=0.5, baseline_ms=2.0, achieved_gbps=400.0, warmup_iterations=2, timed_iterations=3),
            ],
            profile_results=[
                NsightMetrics(occupancy=0.4, regs_per_thread=72),   # sonnet profile
                NsightMetrics(occupancy=0.4, regs_per_thread=72),   # opus profile
            ],
        ),
        store=store,
        cfg=SynthesisConfig(
            retry_budgets=RetryBudgets(performance=1),
            opus_retry_budget_performance=1,
            escalate_to_opus_on_bust=True,
            performance_shape_n=256,
            benchmark_warmup_iterations=2,
            benchmark_timed_iterations=3,
        ),
    )

    result = orchestrator.run(prompt="noop", reference=lambda x: x, target="sm_80")

    assert result.passed is True
    perf_trace = next(t for t in result.report.stage_traces if t.stage_name == "performance")
    # Both models appear in model_used
    assert "claude-sonnet-4-6" in perf_trace.model_used
    assert "claude-opus-4-7" in perf_trace.model_used
