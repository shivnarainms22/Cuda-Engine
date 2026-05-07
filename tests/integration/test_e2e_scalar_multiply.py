import os
import shutil
from pathlib import Path

import pytest

from cuda_engine import SynthesisConfig, synthesize


@pytest.mark.integration
def test_e2e_scalar_multiply_real_llm_and_compile() -> None:
    if "ANTHROPIC_API_KEY" not in os.environ:
        pytest.skip("ANTHROPIC_API_KEY is required for real Stage 2 integration")
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc is required for real Stage 2 integration")

    result = synthesize(
        "Generate a CUDA kernel for scalar multiplication: out = alpha * x "
        "for fp32 tensor x and fp32 scalar alpha.",
        lambda x, alpha: alpha * x,
        "sm_80",
        config=SynthesisConfig(
            artifact_root=".test_artifacts/runs",
            correctness_shapes=((0,), (1,), (127,), (1024,)),
        ),
    )

    run_dir = Path(result.artifacts_dir)
    assert result.passed is True
    assert (run_dir / "stage2_codegen" / "final" / "kernel.cu").exists()
    assert (run_dir / "stage2_codegen" / "final" / "kernel.so").exists()
