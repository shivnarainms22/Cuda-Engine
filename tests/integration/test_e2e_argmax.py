import os
import shutil
from pathlib import Path

import pytest

from cuda_engine import SynthesisConfig, synthesize


@pytest.mark.integration
def test_e2e_argmax_real_llm_and_compile() -> None:
    if "ANTHROPIC_API_KEY" not in os.environ:
        pytest.skip("ANTHROPIC_API_KEY is required for real Stage 2 integration")
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc is required for real Stage 2 integration")

    result = synthesize(
        "Generate a CUDA kernel for fp32 argmax over the last dimension: "
        "out[b] is the int64 index of the maximum value in row b for a contiguous row-major 2D tensor x.",
        lambda x: x.argmax(dim=-1),
        "sm_80",
        config=SynthesisConfig(
            artifact_root=".test_artifacts/runs",
            correctness_shapes=((1, 1), (2, 3), (4, 127), (8, 256)),
            performance_shape_n=16_384,
        ),
    )

    run_dir = Path(result.artifacts_dir)
    assert result.passed is True
    assert (run_dir / "stage2_codegen" / "final" / "kernel.cu").exists()
    assert (run_dir / "stage2_codegen" / "final" / "kernel.so").exists()
