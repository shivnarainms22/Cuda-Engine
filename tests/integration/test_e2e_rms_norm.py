import os
import shutil
from pathlib import Path

import pytest

from cuda_engine import SynthesisConfig, synthesize


@pytest.mark.integration
def test_e2e_rms_norm_fp16_real_llm_and_compile() -> None:
    if "ANTHROPIC_API_KEY" not in os.environ:
        pytest.skip("ANTHROPIC_API_KEY is required for real Stage 2 integration")
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc is required for real Stage 2 integration")

    def rms_norm(x):
        return x * (x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-5).rsqrt().to(x.dtype)

    result = synthesize(
        "Generate a CUDA kernel for fp16 RMSNorm without gamma over the last dimension: "
        "out[b, d] = x[b, d] * rsqrt(mean_d(x[b, d]^2) + 1e-5). "
        "Use fp32 accumulation for the RMS calculation and return fp16 output.",
        rms_norm,
        "sm_80",
        config=SynthesisConfig(
            artifact_root=".test_artifacts/runs",
            correctness_shapes=((1, 16), (2, 64), (4, 128), (8, 256)),
            performance_shape_n=16_384,
        ),
    )

    run_dir = Path(result.artifacts_dir)
    assert result.passed is True
    assert (run_dir / "stage2_codegen" / "final" / "kernel.cu").exists()
    assert (run_dir / "stage2_codegen" / "final" / "kernel.so").exists()
