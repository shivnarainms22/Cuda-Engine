"""Unit tests for evals/kernelbench/filter.py.

Builds a small synthetic "KernelBench-like" tree under tmp_path with a
mix of in-scope, out-of-scope, and needs-review Model files. Verifies
classification logic + report rendering.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from evals.kernelbench.filter import (
    classify_kernelbench_file,
    classify_kernelbench_tree,
    report_to_markdown,
    write_report,
)


def _write_level_file(root: Path, level: str, name: str, body: str) -> Path:
    level_dir = root / level
    level_dir.mkdir(parents=True, exist_ok=True)
    path = level_dir / f"{name}.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


@pytest.fixture
def kb_root(tmp_path: Path) -> Path:
    """Build a fake KernelBench tree with one in-scope, out-of-scope, and review file."""
    root = tmp_path / "KernelBench"
    _write_level_file(
        root,
        "level1",
        "01_gelu",
        """
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return F.gelu(x)
        """,
    )
    _write_level_file(
        root,
        "level1",
        "02_matmul",
        """
        import torch
        import torch.nn as nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b):
                return torch.matmul(a, b)
        """,
    )
    _write_level_file(
        root,
        "level2",
        "03_bias_gelu",
        """
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, bias):
                return F.gelu(x + bias)
        """,
    )
    _write_level_file(
        root,
        "level1",
        "04_attention",
        """
        import torch
        import torch.nn.functional as F

        class Model(torch.nn.Module):
            def forward(self, q, k, v):
                return F.scaled_dot_product_attention(q, k, v)
        """,
    )
    _write_level_file(
        root,
        "level1",
        "05_mystery",
        """
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                # No recognized ops; should require manual review.
                return x
        """,
    )
    # Out-of-scope by level (level3 should not be scanned)
    _write_level_file(
        root,
        "level3",
        "06_resnet",
        """
        import torch.nn as nn

        class Model(nn.Module):
            def forward(self, x):
                return x
        """,
    )
    return root


def test_classify_kernelbench_tree_groups_by_verdict(kb_root: Path) -> None:
    results = classify_kernelbench_tree(kb_root)

    by_name = {r.relative_path: r for r in results}
    # level1/ + level2/ scanned; level3/ skipped entirely.
    assert any("level1" in p for p in by_name)
    assert any("level2" in p for p in by_name)
    assert not any("level3" in p for p in by_name)

    gelu = next(r for r in results if r.relative_path.endswith("01_gelu.py"))
    matmul = next(r for r in results if r.relative_path.endswith("02_matmul.py"))
    bias_gelu = next(r for r in results if r.relative_path.endswith("03_bias_gelu.py"))
    attn = next(r for r in results if r.relative_path.endswith("04_attention.py"))
    mystery = next(r for r in results if r.relative_path.endswith("05_mystery.py"))

    assert gelu.verdict == "in_scope"
    assert "gelu" in gelu.in_scope_ops

    assert matmul.verdict == "out_of_scope"
    assert "matmul" in matmul.out_of_scope_ops

    assert bias_gelu.verdict == "in_scope"
    # bias+gelu shows both add and gelu in its in_scope_ops
    assert "gelu" in bias_gelu.in_scope_ops

    assert attn.verdict == "out_of_scope"
    assert "scaled_dot_product_attention" in attn.out_of_scope_ops

    assert mystery.verdict == "needs_review"
    assert "no recognized" in mystery.reason


def test_classify_kernelbench_file_handles_missing_model_class(tmp_path: Path) -> None:
    path = tmp_path / "no_model.py"
    path.write_text("x = 1\n", encoding="utf-8")

    result = classify_kernelbench_file(path)

    assert result.verdict == "needs_review"
    assert "no Model class" in result.reason


def test_classify_kernelbench_file_handles_syntax_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.py"
    path.write_text("class Model(\n", encoding="utf-8")

    result = classify_kernelbench_file(path)

    assert result.verdict == "needs_review"
    assert "syntax error" in result.reason


def test_classify_kernelbench_file_falls_back_to_first_class_with_forward(tmp_path: Path) -> None:
    """When the class isn't named 'Model', fall back to the first class with forward()."""
    path = tmp_path / "renamed.py"
    path.write_text(
        textwrap.dedent(
            """
            import torch.nn as nn
            import torch.nn.functional as F

            class CustomModel(nn.Module):
                def forward(self, x):
                    return F.relu(x)
            """
        ),
        encoding="utf-8",
    )

    result = classify_kernelbench_file(path)

    assert result.verdict == "in_scope"
    assert result.model_class_name == "CustomModel"
    assert "relu" in result.in_scope_ops


def test_write_report_emits_json_and_markdown(kb_root: Path, tmp_path: Path) -> None:
    results = classify_kernelbench_tree(kb_root)
    out_dir = tmp_path / "report"

    json_path, md_path = write_report(results, out_dir)

    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert len(payload) == len(results)
    assert all("verdict" in entry for entry in payload)

    md_text = md_path.read_text(encoding="utf-8")
    assert "# KernelBench filter report" in md_text
    assert "## in_scope" in md_text
    assert "## out_of_scope" in md_text


def test_report_to_markdown_empty_list_renders_zero_counts() -> None:
    md = report_to_markdown([])
    assert "Total files scanned: 0" in md
    assert "in_scope: 0" in md
    assert "out_of_scope: 0" in md


def test_classify_kernelbench_tree_skips_double_underscore_files(tmp_path: Path) -> None:
    """__init__.py and similar dunder files should not be classified as model files."""
    root = tmp_path / "KernelBench"
    level1 = root / "level1"
    level1.mkdir(parents=True)
    (level1 / "__init__.py").write_text("", encoding="utf-8")

    results = classify_kernelbench_tree(root)

    assert results == []


def test_classify_kernelbench_tree_handles_nested_KernelBench_dir(tmp_path: Path) -> None:
    """Some KernelBench checkouts nest the actual level dirs one level deeper."""
    nested_root = tmp_path / "repo" / "KernelBench"
    nested_root.mkdir(parents=True)
    _write_level_file(
        nested_root,
        "level1",
        "10_relu",
        """
        import torch.nn as nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def forward(self, x):
                return F.relu(x)
        """,
    )

    results = classify_kernelbench_tree(tmp_path / "repo")

    assert len(results) == 1
    assert results[0].verdict == "in_scope"
