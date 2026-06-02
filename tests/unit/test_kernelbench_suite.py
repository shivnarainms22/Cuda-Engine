from pathlib import Path

import yaml

from evals.runner import discover_kernels

EXPECTED_KERNELS = {
    "leaky_relu_fp16",
    "elu_fp16",
    "softplus_fp16",
    "softsign_fp16",
    "mingpt_gelu_fp16",
    "log_softmax_fp16",
    "l1_norm_fp32",
    "argmin_fp32",
    "frobenius_norm_fp32",
    "mse_loss_fp32",
    "reverse_cumsum_fp32",
    "masked_cumsum_fp32",
}

REPO_ROOT = Path(__file__).parents[2]
SUITE_ROOT = REPO_ROOT / "evals" / "kernelbench" / "filtered"
INTERNAL_ROOT = REPO_ROOT / "evals" / "internal"
REQUIRED_FILES = {"prompt.txt", "reference.py", "shapes.yaml", "notes.md"}


def test_kernelbench_suite_has_expected_kernels() -> None:
    kernel_dirs = {path.name for path in SUITE_ROOT.iterdir() if path.is_dir()}

    assert kernel_dirs == EXPECTED_KERNELS


def test_kernelbench_kernels_have_required_files_and_shapes() -> None:
    for kernel_name in sorted(EXPECTED_KERNELS):
        kernel_dir = SUITE_ROOT / kernel_name
        existing_files = {path.name for path in kernel_dir.iterdir() if path.is_file()}
        assert existing_files >= REQUIRED_FILES, kernel_name

        prompt = (kernel_dir / "prompt.txt").read_text(encoding="utf-8").strip()
        notes = (kernel_dir / "notes.md").read_text(encoding="utf-8").strip()
        assert prompt, kernel_name
        # notes.md must cite the KernelBench source for the derivative-work license.
        assert "KernelBench level1/" in notes, kernel_name

        shapes = yaml.safe_load((kernel_dir / "shapes.yaml").read_text(encoding="utf-8"))
        assert isinstance(shapes, list) and len(shapes) >= 3, kernel_name
        for shape in shapes:
            assert isinstance(shape, list) and shape, kernel_name
            assert all(isinstance(dim, int) and dim > 0 for dim in shape), kernel_name


def test_kernelbench_suite_is_discoverable_and_references_import() -> None:
    kernels = discover_kernels(SUITE_ROOT)

    assert {kernel.name for kernel in kernels} == EXPECTED_KERNELS
    assert all(callable(kernel.reference) for kernel in kernels)
    assert all(len(kernel.correctness_shapes) >= 3 for kernel in kernels)


def test_kernelbench_subset_does_not_overlap_internal_suite() -> None:
    """The external subset must be unseen kernels — no name collision with the
    hand-curated internal 30, so it genuinely extends coverage."""
    internal = {path.name for path in INTERNAL_ROOT.iterdir() if path.is_dir()}

    assert EXPECTED_KERNELS.isdisjoint(internal)
