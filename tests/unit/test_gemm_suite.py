"""Structural + sanity tests for the v2.0 GEMM eval suite (Rung 1)."""
from __future__ import annotations

from pathlib import Path

import torch
import yaml

from cuda_engine.cli import _resolve_suite_root
from evals.runner import discover_kernels

REPO = Path(__file__).parents[2]
GEMM_ROOT = REPO / "evals" / "gemm"
INTERNAL_ROOT = REPO / "evals" / "internal"
KERNELBENCH_ROOT = REPO / "evals" / "kernelbench" / "filtered"
REQUIRED_FILES = {"prompt.txt", "reference.py", "shapes.yaml", "notes.md"}

EXPECTED = {"matmul_fp16", "matmul_fp32", "matmul_bias_gelu_fp16"}
# Per-fixture input arg shapes (kind: "nn" = N x N, "n" = length-N) + dtype.
FIXTURE_ARGS = {
    "matmul_fp16": ("fp16", ["nn", "nn"]),
    "matmul_fp32": ("fp32", ["nn", "nn"]),
    "matmul_bias_gelu_fp16": ("fp16", ["nn", "nn", "n"]),
}


def _dirnames(root: Path) -> set[str]:
    return {p.name for p in root.iterdir() if p.is_dir()} if root.exists() else set()


def test_gemm_suite_has_expected_fixtures() -> None:
    assert _dirnames(GEMM_ROOT) == EXPECTED


def test_gemm_fixtures_have_required_files_and_square_shapes() -> None:
    for name in sorted(EXPECTED):
        d = GEMM_ROOT / name
        files = {p.name for p in d.iterdir() if p.is_file()}
        assert files >= REQUIRED_FILES, name
        shapes = yaml.safe_load((d / "shapes.yaml").read_text(encoding="utf-8"))
        assert isinstance(shapes, list) and len(shapes) >= 3, name
        for shape in shapes:
            assert isinstance(shape, list) and len(shape) == 2, name
            assert shape[0] == shape[1] and shape[0] > 0, name  # square


def test_gemm_names_do_not_overlap_internal_or_kernelbench() -> None:
    assert EXPECTED.isdisjoint(_dirnames(INTERNAL_ROOT))
    assert EXPECTED.isdisjoint(_dirnames(KERNELBENCH_ROOT))


def test_gemm_suite_is_discoverable_and_references_callable() -> None:
    kernels = discover_kernels(GEMM_ROOT)
    assert {k.name for k in kernels} == EXPECTED
    assert all(callable(k.reference) for k in kernels)


def _gen(shape: tuple[int, ...], dtype: str, idx: int) -> torch.Tensor:
    n = 1
    for d in shape:
        n *= d
    v = torch.arange(n, dtype=torch.float32).reshape(shape)
    if dtype == "fp16":
        v = (v.remainder(17) - 8) / 8
    return v.to(torch.float16 if dtype == "fp16" else torch.float32) + idx


def test_cli_resolves_gemm_suite_name() -> None:
    assert _resolve_suite_root("gemm") == Path("evals") / "gemm"


def test_gemm_references_produce_finite_outputs() -> None:
    n = 64
    for name, (dtype, kinds) in FIXTURE_ARGS.items():
        m = discover_kernels(GEMM_ROOT)
        ref = next(k.reference for k in m if k.name == name)
        inputs = [
            _gen((n, n) if kind == "nn" else (n,), dtype, idx)
            for idx, kind in enumerate(kinds)
        ]
        out = ref(*inputs)
        assert torch.isfinite(out.float()).all(), name
