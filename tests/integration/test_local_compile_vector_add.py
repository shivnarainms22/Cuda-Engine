import pytest

from cuda_engine.config import SynthesisConfig
from cuda_engine.services.gpu.local import LocalGPURunner


@pytest.mark.integration
def test_local_gpu_runner_can_compile_vector_add_when_nvcc_exists() -> None:
    pytest.importorskip("torch")
    runner = LocalGPURunner(SynthesisConfig(artifact_root=".test_artifacts/gpu"))
    result = runner.compile(
        r'''
extern "C" __global__ void vector_add(const float* x, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + y[i];
}
''',
        target_arch="sm_80",
    )
    if not result.ok and result.errors and "nvcc not found" in result.errors[0]:
        pytest.skip("nvcc not available")
    assert result.ok, result.log
    assert result.so_path is not None
    assert result.so_path.exists()
