"""Tests for orchestrator checkpoint writing (Task 3) and resume (Task 4)."""
from __future__ import annotations

from pathlib import Path

import pytest

from cuda_engine.config import RetryBudgets, SynthesisConfig
from cuda_engine.orchestrator import Orchestrator
from cuda_engine.services.gpu.base import BenchmarkResult, CompileResult, RunResult
from cuda_engine.services.gpu.mocks import MockGPURunner
from cuda_engine.services.llm.base import LLMResponse
from cuda_engine.services.llm.mocks import MockLLMClient
from cuda_engine.services.store.mocks import InMemoryStore

# Minimal spec JSON (identity kernel, same as test_orchestrator.py)
SPEC_JSON = (
    '{"name":"identity","target_arch":"sm_80",'
    '"inputs":[{"name":"x","dtype":"fp32","shape":["N"]}],'
    '"outputs":[{"name":"out","dtype":"fp32","shape":["N"]}],'
    '"precision_tolerance":{"rtol":0.001,"atol":0.001},'
    '"optimization_priority":"balanced"}'
)

SHAPE_SIZES = (0, 1, 127, 128, 1024, 4097)


def _identity(x: object) -> object:  # stable reference callable
    return x


def _codegen_response(src: str = "extern code") -> LLMResponse:
    return LLMResponse(
        text=f"```cuda\n{src}\n```",
        model="mock",
        tool_calls=[
            {"name": "compile_kernel", "input": {"src": src, "target_arch": "sm_80"}}
        ],
    )


def _make_correctness_run_results() -> list[RunResult]:
    torch = __import__("torch")
    return [
        RunResult(ok=True, output_tensors=[torch.arange(size, dtype=torch.float32)])
        for size in SHAPE_SIZES
    ]


def _happy_path_orchestrator(store: InMemoryStore) -> Orchestrator:
    """Minimal happy-path orchestrator (no perf retries, passes correctness)."""
    torch = __import__("torch")
    return Orchestrator(
        llm=MockLLMClient(
            responses=[
                SPEC_JSON,
                _codegen_response(),
                "```cuda\n// annotated\n```",
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
        cfg=SynthesisConfig(retry_budgets=RetryBudgets(performance=0)),
    )


# ---------------------------------------------------------------------------
# Task 3: checkpoint.json written after each stable boundary
# ---------------------------------------------------------------------------


def test_checkpoint_written_after_full_run() -> None:
    """Full happy-path run leaves checkpoint with all 4 stages and all 4 objects."""
    store = InMemoryStore()
    orc = _happy_path_orchestrator(store)
    result = orc.run(prompt="noop", reference=lambda x: x, target="sm_80")
    assert result.passed is True

    cp = store.read_json(result.run_id, "checkpoint.json")
    assert cp["completed_stages"] == ["interview", "correctness", "performance", "polish"]
    objects = cp["objects"]
    assert "spec" in objects
    assert "artifact" in objects
    assert "correctness" in objects
    assert "performance" in objects


def test_checkpoint_has_valid_fingerprint() -> None:
    """Checkpoint stores a valid sha256 inputs fingerprint."""
    store = InMemoryStore()
    orc = _happy_path_orchestrator(store)
    result = orc.run(prompt="noop", reference=_identity, target="sm_80")

    cp = store.read_json(result.run_id, "checkpoint.json")
    assert isinstance(cp["inputs_fingerprint"], str)
    assert len(cp["inputs_fingerprint"]) == 64  # sha256 hex


# ---------------------------------------------------------------------------
# Task 4: resume_run_id
# ---------------------------------------------------------------------------


def test_resume_missing_checkpoint() -> None:
    """resume_run_id for a non-existent run raises a clear error."""
    store = InMemoryStore()
    orc = Orchestrator(
        llm=MockLLMClient(responses=[]),
        gpu=MockGPURunner(),
        store=store,
        cfg=SynthesisConfig(),
    )
    with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
        orc.run(
            prompt="noop",
            reference=lambda x: x,
            target="sm_80",
            resume_run_id="nonexistent_run_id",
        )


def test_resume_fingerprint_mismatch() -> None:
    """Resuming with a different prompt raises with a message about inputs/fingerprint."""
    store = InMemoryStore()
    orc = _happy_path_orchestrator(store)
    result = orc.run(prompt="original prompt", reference=lambda x: x, target="sm_80")
    run_id = result.run_id

    fresh_orc = Orchestrator(
        llm=MockLLMClient(responses=[]),
        gpu=MockGPURunner(),
        store=store,
        cfg=SynthesisConfig(),
    )
    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        fresh_orc.run(
            prompt="DIFFERENT PROMPT",
            reference=lambda x: x,
            target="sm_80",
            resume_run_id=run_id,
        )
    msg = str(exc_info.value).lower()
    assert any(kw in msg for kw in ("fingerprint", "inputs", "changed", "mismatch", "cannot resume"))


def test_resume_skips_completed_stages() -> None:
    """Crash during performance (after correctness checkpointed); resume issues zero
    LLM calls for interview/codegen/correctness and completes with only perf+polish calls."""
    torch = __import__("torch")
    store = InMemoryStore()

    # --- First run: crashes during performance retry (LLM runs dry) ---
    crash_orc = Orchestrator(
        llm=MockLLMClient(responses=[SPEC_JSON, _codegen_response()]),  # no perf/polish responses
        gpu=MockGPURunner(
            compile_results=[CompileResult(ok=True, so_path=Path("kernel.so"), log="ok")],
            run_results=[
                RunResult(ok=True, output_tensors=[torch.arange(size, dtype=torch.float32)])
                for size in SHAPE_SIZES
            ],
            # Initial benchmark below target so retry loop is entered (and LLM called)
            benchmark_results=[
                BenchmarkResult(ok=True, custom_ms=2.0, baseline_ms=1.0),
            ],
        ),
        store=store,
        cfg=SynthesisConfig(
            retry_budgets=RetryBudgets(performance=1),
            perf_target_speedup_vs_torch_compile=1.0,
        ),
    )

    with pytest.raises(RuntimeError, match="no canned responses"):
        crash_orc.run(prompt="noop", reference=_identity, target="sm_80")

    # Get run_id from the checkpoint written before the crash
    run_ids = {key[0] for key in store._files if key[1] == "checkpoint.json"}
    assert len(run_ids) == 1, f"expected exactly 1 run, got {run_ids}"
    run_id = next(iter(run_ids))

    # Checkpoint should have interview+correctness but NOT performance
    cp_data = store.read_json(run_id, "checkpoint.json")
    assert "interview" in cp_data["completed_stages"]
    assert "correctness" in cp_data["completed_stages"]
    assert "performance" not in cp_data["completed_stages"]
    assert "polish" not in cp_data["completed_stages"]

    # --- Resume with a FRESH LLM that only has performance+polish responses ---
    resume_llm = MockLLMClient(
        responses=[
            _codegen_response("fixed perf"),        # performance retry: new kernel
            "```cuda\n// annotated\n```",            # polish
        ]
    )
    resume_orc = Orchestrator(
        llm=resume_llm,
        gpu=MockGPURunner(
            compile_results=[
                CompileResult(ok=True, so_path=Path("perf_fixed.so"), log="ok"),  # perf fix
                CompileResult(ok=True, so_path=Path("polish.so"), log="ok"),       # polish
            ],
            run_results=[
                RunResult(ok=True, output_tensors=[torch.arange(size, dtype=torch.float32)])
                for size in SHAPE_SIZES
            ],
            benchmark_results=[
                # Initial benchmark on resumed artifact → below target (triggers retry)
                BenchmarkResult(ok=True, custom_ms=2.0, baseline_ms=1.0),
                # After perf fix → above target
                BenchmarkResult(ok=True, custom_ms=0.5, baseline_ms=1.0),
            ],
        ),
        store=store,
        cfg=SynthesisConfig(
            retry_budgets=RetryBudgets(performance=1),
            perf_target_speedup_vs_torch_compile=1.0,
        ),
    )

    result = resume_orc.run(
        prompt="noop",
        reference=_identity,
        target="sm_80",
        resume_run_id=run_id,
    )

    assert result.passed is True
    assert result.run_id == run_id
    # Interview/codegen/correctness issued ZERO new LLM calls on resume
    assert resume_llm.call_count == 2, (
        f"expected 2 LLM calls (perf+polish), got {resume_llm.call_count}"
    )

    # Final checkpoint should have all 4 stages completed
    final_cp = store.read_json(run_id, "checkpoint.json")
    assert final_cp["completed_stages"] == ["interview", "correctness", "performance", "polish"]
