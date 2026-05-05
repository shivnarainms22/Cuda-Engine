import shutil
from pathlib import Path

import pytest

from cuda_engine.config import SynthesisConfig
from cuda_engine.services.gpu.local import LocalGPURunner


@pytest.mark.integration
def test_local_gpu_runner_runs_known_good_torch_custom_op() -> None:
    torch = pytest.importorskip("torch")
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    runner = LocalGPURunner(SynthesisConfig(artifact_root=".test_artifacts/gpu"))
    compile_result = runner.compile(_custom_op_vector_add_src(), target_arch="sm_80")
    assert compile_result.ok, compile_result.log
    assert compile_result.so_path is not None

    x = torch.arange(16, device="cuda", dtype=torch.float32)
    y = torch.ones(16, device="cuda", dtype=torch.float32)
    run_result = runner.run_kernel(Path(compile_result.so_path), [x, y], timeout_seconds=30)

    assert run_result.ok, run_result.stderr
    assert run_result.output_tensors is not None
    torch.testing.assert_close(run_result.output_tensors[0], x + y)

    benchmark = runner.benchmark_kernel(
        Path(compile_result.so_path),
        [x, y],
        warmup_iterations=2,
        timed_iterations=5,
        timeout_seconds=30,
    )
    assert benchmark.ok, benchmark.stderr
    assert benchmark.custom_ms > 0
    assert benchmark.baseline_ms is not None
    assert benchmark.baseline_ms > 0


def _custom_op_vector_add_src() -> str:
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

TORCH_LIBRARY(cuda_engine, m) {
    m.def("forward(Tensor x, Tensor y) -> Tensor");
}

TORCH_LIBRARY_IMPL(cuda_engine, CUDA, m) {
    m.impl("forward", &forward);
}
'''
