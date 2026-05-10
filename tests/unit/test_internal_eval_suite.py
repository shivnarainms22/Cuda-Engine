from pathlib import Path

import yaml

from evals.runner import discover_kernels

EXPECTED_KERNELS = {
    "vector_add_fp32",
    "scalar_multiply_fp32",
    "rms_norm_fp16",
    "sum_reduction_fp32",
    "argmax_fp32",
    "layernorm_fp16",
    "silu_fp16",
    "gelu_fp16",
    "swiglu_fp16",
    "geglu_fp16",
    "relu_bias_fp32",
    "dropout_fp16",
    "softmax_lastdim_fp16",
    "rmsnorm_silu_fused_fp16",
    "layernorm_silu_fused_fp16",
    "bias_gelu_fp16",
    "add_relu_fp32",
    "sigmoid_mul_fp16",
    "tanh_add_fp32",
    "clamp_fp32",
    "mean_lastdim_fp32",
    "max_lastdim_fp32",
    "min_lastdim_fp32",
    "topk_fp32",
    "prefix_sum_fp32",
    "segment_sum_fp32",
    "masked_mean_fp16",
    "softmax_numerator_fp16",
    "cumulative_max_fp32",
    "l2_norm_fp32",
}

SUITE_ROOT = Path(__file__).parents[2] / "evals" / "internal"
REQUIRED_FILES = {"prompt.txt", "reference.py", "shapes.yaml", "notes.md"}


def test_internal_eval_suite_has_expected_30_kernels() -> None:
    kernel_dirs = {path.name for path in SUITE_ROOT.iterdir() if path.is_dir()}

    assert kernel_dirs == EXPECTED_KERNELS


def test_internal_eval_kernels_have_required_files_and_shapes() -> None:
    for kernel_name in sorted(EXPECTED_KERNELS):
        kernel_dir = SUITE_ROOT / kernel_name
        existing_files = {path.name for path in kernel_dir.iterdir() if path.is_file()}
        assert existing_files >= REQUIRED_FILES, kernel_name

        prompt = (kernel_dir / "prompt.txt").read_text(encoding="utf-8").strip()
        notes = (kernel_dir / "notes.md").read_text(encoding="utf-8").strip()
        assert prompt, kernel_name
        assert notes, kernel_name

        shapes = yaml.safe_load((kernel_dir / "shapes.yaml").read_text(encoding="utf-8"))
        assert isinstance(shapes, list), kernel_name
        assert len(shapes) >= 3, kernel_name
        for shape in shapes:
            assert isinstance(shape, list), kernel_name
            assert shape
            assert all(isinstance(dim, int) and dim > 0 for dim in shape), kernel_name


def test_internal_eval_suite_is_discoverable_and_references_import() -> None:
    kernels = discover_kernels(SUITE_ROOT)

    assert {kernel.name for kernel in kernels} == EXPECTED_KERNELS
    assert len(kernels) == 30
    assert all(callable(kernel.reference) for kernel in kernels)
    assert all(len(kernel.correctness_shapes) >= 3 for kernel in kernels)


def test_layernorm_silu_fixture_disambiguates_no_affine_inputs() -> None:
    prompt = (SUITE_ROOT / "layernorm_silu_fused_fp16" / "prompt.txt").read_text(
        encoding="utf-8"
    )

    assert "exactly one input tensor x" in prompt
    assert "Do not use affine gamma or beta" in prompt
