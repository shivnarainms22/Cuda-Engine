"""Integration test for torch.compile baseline measurement.

Gated on real CUDA + nvcc. Confirms LocalGPURunner.benchmark_kernel
captures a real torch.compile baseline for a known reference function,
and that intentional failures surface as baseline_error without
breaking the custom kernel benchmark.
"""

import shutil

import pytest

from cuda_engine.config import SynthesisConfig
from cuda_engine.services.gpu.local import LocalGPURunner


def _relu_reference(x):
    """Top-level def so it pickles cleanly into the subprocess child."""
    import torch
    return torch.relu(x)


def _bad_reference(x):
    """A reference that raises in torch.compile to test failure surfacing."""
    raise RuntimeError("intentional baseline failure for test")


@pytest.mark.integration
def test_baseline_torch_compile_succeeds_against_real_gpu() -> None:
    pytest.importorskip("torch")
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc not available")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    runner = LocalGPURunner(SynthesisConfig(artifact_root=".test_artifacts/baseline"))
    compile_result = runner.compile(_torch_extension_relu(), target_arch="sm_80")
    assert compile_result.ok, compile_result.log
    assert compile_result.so_path is not None

    x = torch.randn(1 << 14, device="cuda", dtype=torch.float32)
    result = runner.benchmark_kernel(
        compile_result.so_path,
        [x],
        reference=_relu_reference,
        warmup_iterations=3,
        timed_iterations=20,
    )

    assert result.ok, result.stderr
    assert result.custom_ms > 0
    assert result.baseline_ms is not None, f"baseline failed: {result.baseline_error}"
    assert result.baseline_error is None
    assert 1e-3 <= result.baseline_ms <= 100.0


@pytest.mark.integration
def test_baseline_torch_compile_failure_surfaces_error() -> None:
    pytest.importorskip("torch")
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc not available")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    runner = LocalGPURunner(SynthesisConfig(artifact_root=".test_artifacts/baseline_fail"))
    compile_result = runner.compile(_torch_extension_relu(), target_arch="sm_80")
    assert compile_result.ok

    x = torch.randn(1 << 14, device="cuda", dtype=torch.float32)
    result = runner.benchmark_kernel(
        compile_result.so_path,
        [x],
        reference=_bad_reference,
        warmup_iterations=2,
        timed_iterations=5,
    )

    assert result.ok, "custom kernel must still benchmark even if baseline fails"
    assert result.custom_ms > 0
    assert result.baseline_ms is None
    assert result.baseline_error is not None
    assert "torch.compile baseline failed" in result.baseline_error
    assert "RuntimeError" in result.baseline_error
    assert "intentional baseline failure" in result.baseline_error


def _torch_extension_relu() -> str:
    return r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void relu_kernel(const float* x, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

torch::Tensor forward(torch::Tensor x) {
    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n
    );
    return out;
}

TORCH_LIBRARY(cuda_engine, m) { m.def("forward(Tensor x) -> Tensor"); }
TORCH_LIBRARY_IMPL(cuda_engine, CUDA, m) { m.impl("forward", &forward); }
'''
