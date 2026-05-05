from pathlib import Path

from cuda_engine.services.gpu.base import BenchmarkResult, CompileResult, RunResult
from cuda_engine.services.gpu.mocks import MockGPURunner


def test_mock_gpu_compile_canned_results() -> None:
    canned = CompileResult(ok=True, so_path=Path("/tmp/x.so"), log="ok", ptx_size_bytes=42)
    mock = MockGPURunner(compile_results=[canned])

    result = mock.compile("kernel src", target_arch="sm_80")

    assert result.ok
    assert result.ptx_size_bytes == 42


def test_mock_gpu_run_kernel() -> None:
    canned = RunResult(ok=True, stdout="", wall_seconds=0.001)
    mock = MockGPURunner(run_results=[canned])

    result = mock.run_kernel(Path("/tmp/x.so"), inputs=[])

    assert result.ok


def test_mock_gpu_benchmark_kernel() -> None:
    canned = BenchmarkResult(ok=True, custom_ms=0.5, baseline_ms=1.0)
    mock = MockGPURunner(benchmark_results=[canned])

    result = mock.benchmark_kernel(Path("/tmp/x.so"), inputs=[])

    assert result.ok
    assert result.custom_ms == 0.5
    assert result.baseline_ms == 1.0
