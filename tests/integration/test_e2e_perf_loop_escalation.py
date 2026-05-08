import os
import shutil
from pathlib import Path

import pytest

from cuda_engine import SynthesisConfig, synthesize
from cuda_engine.config import RetryBudgets


def _slow_softmax_reference(x):
    """Reference for a deliberately optimization-rich kernel.

    Softmax over the last dim. The naive CUDA implementation (per-row
    two-pass without shared-memory tiling or warp reductions) typically
    lands well below 1.0x torch.compile on A100, exercising the perf
    retry loop. Opus then either tiles, uses warp-level reductions, or
    fuses passes to cross the bar.
    """
    import torch

    return torch.softmax(x, dim=-1)


@pytest.mark.integration
def test_perf_loop_escalates_to_opus_on_softmax_e2e(tmp_path: Path) -> None:
    """End-to-end: real Anthropic + A100 + ncu, verify Stage 4 escalation runs.

    Assertions are deliberately soft. We require the escalation path to RUN
    (both Sonnet and Opus touched the perf stage), not that Opus actually
    beat the target. Whether Opus converges on a >=1.0x kernel for this
    contrived workload is real-world variance not pinnable in CI.
    """
    pytest.importorskip("torch")
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc not available")
    if shutil.which("ncu") is None:
        pytest.skip("ncu not available")
    if "ANTHROPIC_API_KEY" not in os.environ:
        pytest.skip("ANTHROPIC_API_KEY is required for real Stage 4 escalation")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cfg = SynthesisConfig(
        artifact_root=str(tmp_path),
        retry_budgets=RetryBudgets(codegen=2, correctness=2, performance=1),
        opus_retry_budget_performance=1,
        opus_retry_budget_codegen=1,
        performance_shape_n=4096,
        benchmark_warmup_iterations=3,
        benchmark_timed_iterations=10,
    )

    result = synthesize(
        prompt="Compute softmax over the last dim of a 2D float32 tensor.",
        reference=_slow_softmax_reference,
        target="sm_80",
        config=cfg,
    )

    assert result.run_id

    perf_trace = next(
        (t for t in result.report.stage_traces if t.stage_name == "performance"),
        None,
    )
    assert perf_trace is not None, "performance stage missing from trace"

    if "claude-opus-4-7" not in perf_trace.model_used:
        pytest.skip(
            f"escalation did not trigger (Sonnet hit target on first try): "
            f"model_used={perf_trace.model_used}"
        )

    assert "claude-sonnet-4-6" in perf_trace.model_used
    assert "claude-opus-4-7" in perf_trace.model_used

    run_dir = Path(result.artifacts_dir)
    sonnet_attempt = run_dir / "stage4_performance" / "perf_repair" / "attempt_01"
    opus_attempt = run_dir / "stage4_performance" / "perf_repair" / "attempt_02"
    assert sonnet_attempt.exists(), f"missing {sonnet_attempt}"
    assert opus_attempt.exists(), f"missing {opus_attempt}"
    assert (opus_attempt / "nsight.json").exists()
    assert (opus_attempt / "benchmark.json").exists()

    notes_str = " ".join(result.performance.notes) if result.performance else ""
    assert "escalated to opus" in notes_str.lower()
