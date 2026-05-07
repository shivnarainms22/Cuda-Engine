import json
from pathlib import Path

from cuda_engine.config import RetryBudgets, SynthesisConfig
from cuda_engine.orchestrator import Orchestrator
from cuda_engine.services.gpu.base import CompileResult, RunResult
from cuda_engine.services.gpu.mocks import MockGPURunner
from cuda_engine.services.llm.base import LLMResponse
from cuda_engine.services.llm.mocks import MockLLMClient
from cuda_engine.services.store.mocks import InMemoryStore

SPEC_JSON = """{"name":"identity","target_arch":"sm_80","inputs":[{"name":"x","dtype":"fp32","shape":["N"]}],"outputs":[{"name":"out","dtype":"fp32","shape":["N"]}],"precision_tolerance":{"rtol":0.001,"atol":0.001},"optimization_priority":"balanced"}"""

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
