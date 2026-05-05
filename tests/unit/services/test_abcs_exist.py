import inspect

from cuda_engine.services.gpu.base import CompileResult, GPURunner, NsightMetrics, RunResult
from cuda_engine.services.llm.base import LLMClient, LLMResponse, ToolSpec
from cuda_engine.services.store.base import ArtifactStore


def test_llm_client_is_abstract() -> None:
    assert inspect.isabstract(LLMClient)
    assert "complete" in dir(LLMClient)
    assert ToolSpec
    assert LLMResponse


def test_gpu_runner_is_abstract() -> None:
    assert inspect.isabstract(GPURunner)
    for method_name in ("compile", "run_kernel", "profile"):
        assert method_name in dir(GPURunner)
    assert CompileResult
    assert RunResult
    assert NsightMetrics


def test_store_is_abstract() -> None:
    assert inspect.isabstract(ArtifactStore)
    for method_name in ("new_run", "write_text", "write_bytes", "write_json", "run_dir"):
        assert method_name in dir(ArtifactStore)
