"""Tests for the cross-provider comparison report (v1.1 spec §2.4, deferred Task 12)."""
from __future__ import annotations

from pathlib import Path

from evals.runner import (
    EvalRow,
    _write_csv,
    build_provider_comparison,
    read_rows_from_csv,
)


def _row(model_id: str, provider: str, passed: bool, speedup: float | None, kernel: str) -> EvalRow:
    return EvalRow(
        kernel=kernel,
        passed=passed,
        run_id="r",
        failed_stage=None,
        failure_reason="",
        speedup_vs_torch_compile=speedup,
        speedup_vs_reference=None,
        below_target=(speedup is not None and speedup < 1.0),
        artifacts_dir="/a",
        provider=provider,
        model_id=model_id,
    )


def test_build_provider_comparison_groups_by_model_and_reports_metrics() -> None:
    rows = [
        _row("anthropic:claude-sonnet-4-6", "anthropic", True, 1.2, "k1"),
        _row("anthropic:claude-sonnet-4-6", "anthropic", True, 0.8, "k2"),
        _row("openai:gpt-4o", "openai", False, None, "k1"),
        _row("openai:gpt-4o", "openai", True, 1.5, "k2"),
    ]
    md = build_provider_comparison(rows)
    assert "# Provider comparison" in md
    assert "anthropic:claude-sonnet-4-6" in md
    assert "openai:gpt-4o" in md
    assert "2/2 (100%)" in md   # anthropic functional
    assert "1/2 (50%)" in md    # openai functional
    # exactly one table row per model id
    assert md.count("anthropic:claude-sonnet-4-6") == 1
    assert md.count("openai:gpt-4o") == 1


def test_build_provider_comparison_handles_unspecified_model() -> None:
    rows = [_row("", "", True, 1.0, "k1")]
    md = build_provider_comparison(rows)
    assert "(unspecified)" in md


def test_read_rows_from_csv_roundtrips_provider_and_model(tmp_path: Path) -> None:
    rows = [
        _row("openai:gpt-4o", "openai", True, 1.5, "k1"),
        _row("openai:gpt-4o", "openai", False, None, "k2"),
    ]
    csv_path = tmp_path / "results.csv"
    _write_csv(csv_path, rows)

    back = read_rows_from_csv(csv_path)
    assert [r.kernel for r in back] == ["k1", "k2"]
    assert back[0].model_id == "openai:gpt-4o"
    assert back[0].provider == "openai"
    assert back[0].passed is True
    assert back[0].speedup_vs_torch_compile == 1.5
    assert back[1].passed is False
    assert back[1].speedup_vs_torch_compile is None


def test_read_then_compare_end_to_end(tmp_path: Path) -> None:
    """Read two per-provider results.csv files and produce a combined comparison."""
    a = tmp_path / "anthropic.csv"
    b = tmp_path / "openai.csv"
    _write_csv(a, [_row("anthropic:x", "anthropic", True, 1.1, "k1")])
    _write_csv(b, [_row("openai:y", "openai", True, 2.0, "k1")])
    rows = read_rows_from_csv(a) + read_rows_from_csv(b)
    md = build_provider_comparison(rows)
    assert "anthropic:x" in md and "openai:y" in md


def test_cli_compare_providers_reads_dirs_and_writes_markdown(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from cuda_engine.cli import app

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _write_csv(tmp_path / "a" / "results.csv", [_row("anthropic:x", "anthropic", True, 1.1, "k1")])
    _write_csv(tmp_path / "b" / "results.csv", [_row("openai:y", "openai", True, 2.0, "k1")])
    out = tmp_path / "cmp.md"

    result = CliRunner().invoke(
        app,
        ["compare-providers", str(tmp_path / "a"), str(tmp_path / "b"), "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "anthropic:x" in text and "openai:y" in text
