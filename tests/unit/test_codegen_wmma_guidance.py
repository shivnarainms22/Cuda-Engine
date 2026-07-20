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


def test_guidance_matches_row_major_input_layout() -> None:
    """Inputs are row-major contiguous (torch .reshape) and the fixture prompts say
    so. A col_major matrix_b fragment silently computes A @ B.T — it COMPILES, so it
    burns correctness/repair attempts rather than failing fast."""
    g = _tensor_core_matmul_guidance(_MM_FP16)
    assert g is not None
    assert "wmma::matrix_b, 16, 16, 16, half, wmma::row_major" in g
    # No col_major fragment may be DECLARED; naming it in the prose warning is fine.
    assert "wmma::col_major" not in g


def test_guidance_requires_handling_non_multiple_of_16_shapes() -> None:
    """The hard correctness gate runs shapes that are not multiples of 16, so a
    WMMA-only kernel cannot pass. The guidance must not present the remainder path
    as optional. Derived from config so this fails if correctness_shapes changes."""
    from cuda_engine.config import SynthesisConfig

    ragged = [n for (n, *_rest) in SynthesisConfig().correctness_shapes if n % 16 and n > 1]
    assert ragged, "expected at least one non-multiple-of-16 correctness shape"

    g = _tensor_core_matmul_guidance(_MM_FP16)
    assert g is not None
    for n in ragged:
        assert str(n) in g, f"guidance never mentions correctness shape {n}"
    assert "only add a scalar remainder path if" not in g


def test_guidance_shows_fp32_accumulator_to_fp16_output_conversion() -> None:
    """store_matrix_sync requires the pointer type to match the fragment type, so a
    float accumulator cannot be stored straight to the half* output of an fp16
    matmul — that is a hard compile error. The guidance must show the conversion."""
    g = _tensor_core_matmul_guidance(_MM_FP16)
    assert g is not None
    assert "__float2half" in g
    assert "float" in g and "store_matrix_sync" in g


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
