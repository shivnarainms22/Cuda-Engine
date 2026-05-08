import shutil

import pytest

from cuda_engine.config import SynthesisConfig
from cuda_engine.services.gpu.local import LocalGPURunner


@pytest.mark.integration
def test_local_gpu_runner_profile_collects_real_ncu_metrics() -> None:
    torch = pytest.importorskip("torch")
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc not available")
    if shutil.which("ncu") is None:
        pytest.skip("ncu not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    runner = LocalGPURunner(SynthesisConfig(artifact_root=".test_artifacts/gpu"))
    compile_result = runner.compile(_torch_extension_vector_add(), target_arch="sm_80")
    assert compile_result.ok, compile_result.log
    assert compile_result.so_path is not None

    x = torch.arange(1 << 14, device="cuda", dtype=torch.float32)
    y = torch.ones_like(x)

    metrics = runner.profile(compile_result.so_path, [x, y])

    if metrics.raw_csv.startswith("ERR_NVGPUCTRPERM") or "permission" in metrics.raw_csv.lower():
        pytest.skip(f"ncu lacks profiling permission on this host: {metrics.raw_csv[:200]}")

    assert metrics.occupancy is not None, metrics.raw_csv[:500]
    assert 0.0 < metrics.occupancy <= 1.0
    assert metrics.regs_per_thread is not None
    assert metrics.regs_per_thread > 0


def _torch_extension_vector_add() -> str:
    return r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void vector_add_kernel(const float* x, const float* y, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = x[i] + y[i];
    }
}

torch::Tensor forward(torch::Tensor x, torch::Tensor y) {
    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), n
    );
    return out;
}

TORCH_LIBRARY(cuda_engine_profile, m) {
    m.def("forward(Tensor x, Tensor y) -> Tensor");
}

TORCH_LIBRARY_IMPL(cuda_engine_profile, CUDA, m) {
    m.impl("forward", &forward);
}
'''
