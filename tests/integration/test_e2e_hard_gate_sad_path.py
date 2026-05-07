import os
import shutil
from pathlib import Path

import pytest

from cuda_engine import SynthesisConfig, synthesize
from cuda_engine.config import RetryBudgets


@pytest.mark.integration
def test_e2e_hard_gate_fails_when_reference_disagrees_with_prompt() -> None:
    """M2 hard gate evidence on real GPU.

    Prompt asks for elementwise add (x + y); reference returns 2 * x. The
    kernel synthesized from the prompt cannot match the reference, so Stage 3
    must fail. With correctness retry budget = 0, repair is disabled and the
    failure surfaces immediately.
    """
    if "ANTHROPIC_API_KEY" not in os.environ:
        pytest.skip("ANTHROPIC_API_KEY is required for real Stage 2 integration")
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc is required for real Stage 2 integration")

    result = synthesize(
        "Generate a CUDA kernel for vector addition: out = x + y for fp32 tensors.",
        lambda x, y: 2 * x,
        "sm_80",
        config=SynthesisConfig(
            artifact_root=".test_artifacts/runs",
            retry_budgets=RetryBudgets(correctness=0),
        ),
    )

    run_dir = Path(result.artifacts_dir)
    assert result.passed is False
    assert result.failed_stage == 3
    assert result.correctness is not None
    assert result.correctness.passed is False
    assert result.correctness.failing_inputs, "expected at least one failing-input record"
    assert (run_dir / "stage3_correctness" / "report.json").exists()
    assert (run_dir / "report.json").exists()
