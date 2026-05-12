import inspect

from cuda_engine.services.gpu.base import (
    BenchmarkResult,
    CompileResult,
    GPURunner,
    NsightMetrics,
    RunResult,
)
from cuda_engine.services.llm.base import LLMClient, LLMResponse, ToolSpec
from cuda_engine.services.store.base import ArtifactStore


def test_llm_client_is_abstract() -> None:
    assert inspect.isabstract(LLMClient)
    assert "complete" in dir(LLMClient)
    assert ToolSpec
    assert LLMResponse


def test_gpu_runner_is_abstract() -> None:
    assert inspect.isabstract(GPURunner)
    for method_name in ("compile", "run_kernel", "benchmark_kernel", "profile"):
        assert method_name in dir(GPURunner)
    assert CompileResult
    assert RunResult
    assert BenchmarkResult
    assert NsightMetrics


def test_store_is_abstract() -> None:
    assert inspect.isabstract(ArtifactStore)
    for method_name in ("new_run", "write_text", "write_bytes", "write_json", "run_dir"):
        assert method_name in dir(ArtifactStore)


def test_benchmark_result_carries_baseline_error_field() -> None:
    """Failed torch.compile baseline must surface as a structured error string."""
    result = BenchmarkResult(
        ok=True,
        custom_ms=0.01,
        baseline_ms=None,
        baseline_error="torch.compile baseline failed: RuntimeError: graph too large",
    )

    assert result.baseline_error == "torch.compile baseline failed: RuntimeError: graph too large"
    assert BenchmarkResult(ok=True, custom_ms=0.01).baseline_error is None
