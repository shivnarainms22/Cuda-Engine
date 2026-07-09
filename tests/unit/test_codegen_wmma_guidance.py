"""Rung 4: WMMA/tensor-core codegen guidance injected for fp16/bf16 matmul specs."""
from __future__ import annotations

from pathlib import Path

from cuda_engine.models import (
    KernelSpec,
    OptimizationPriority,
    PrecisionTolerance,
    TensorArg,
)
from cuda_engine.services.gpu.base import CompileResult
from cuda_engine.services.gpu.mocks import MockGPURunner
from cuda_engine.services.llm.base import LLMResponse
from cuda_engine.services.llm.mocks import MockLLMClient
from cuda_engine.services.store.mocks import InMemoryStore
from cuda_engine.stages.codegen import (
    Stage2Codegen,
    _is_matmul_spec,
    _tensor_core_matmul_guidance,
)


def _spec(inputs: list[TensorArg], out_dtype: str = "fp16") -> KernelSpec:
    return KernelSpec(
        name="k",
        target_arch="sm_80",
        inputs=inputs,
        outputs=[TensorArg(name="o", dtype=out_dtype, shape=("M", "N"))],
        precision_tolerance=PrecisionTolerance(),
        optimization_priority=OptimizationPriority.THROUGHPUT,
    )


_MM_FP16 = _spec([
    TensorArg(name="a", dtype="fp16", shape=("M", "K")),
    TensorArg(name="b", dtype="fp16", shape=("K", "N")),
])
_ELEMENTWISE = _spec([TensorArg(name="x", dtype="fp16", shape=("N",))])
_MM_FP32 = _spec(
    [
        TensorArg(name="a", dtype="fp32", shape=("M", "K")),
        TensorArg(name="b", dtype="fp32", shape=("K", "N")),
    ],
    out_dtype="fp32",
)


def test_is_matmul_spec_needs_two_rank2_inputs() -> None:
    assert _is_matmul_spec(_MM_FP16) is True
    assert _is_matmul_spec(_ELEMENTWISE) is False


def test_guidance_only_for_tensor_core_matmul_dtypes() -> None:
    g = _tensor_core_matmul_guidance(_MM_FP16)
    assert g is not None and "wmma::fragment" in g and "mma_sync" in g
    assert _tensor_core_matmul_guidance(_ELEMENTWISE) is None  # not a matmul
    assert _tensor_core_matmul_guidance(_MM_FP32) is None       # fp32 input isn't a TC dtype key


def _compiling_response() -> LLMResponse:
    return LLMResponse(
        text="```cuda\ncode\n```",
        model="mock",
        tool_calls=[{"name": "compile_kernel", "input": {"src": "code", "target_arch": "sm_80"}}],
    )


def _run_and_capture_system(spec: KernelSpec) -> str:
    llm = MockLLMClient([_compiling_response()])
    gpu = MockGPURunner(compile_results=[CompileResult(ok=True, so_path=Path("k.so"), log="ok")])
    Stage2Codegen(llm=llm, gpu=gpu, store=InMemoryStore()).run(spec=spec, run_id="r", model="mock")
    return " ".join(block["text"] for block in llm.calls[0]["system"])


def test_wmma_guidance_injected_for_matmul_fp16() -> None:
    assert "wmma::fragment" in _run_and_capture_system(_MM_FP16)


def test_wmma_guidance_absent_for_elementwise() -> None:
    assert "wmma::fragment" not in _run_and_capture_system(_ELEMENTWISE)
