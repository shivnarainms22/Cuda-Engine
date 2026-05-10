import json
import os
import uuid
from pathlib import Path

from typer.testing import CliRunner

from cuda_engine.cli import app
from evals.runner import EvalRow, EvalSummary


def test_show_report_prints_compact_success_summary() -> None:
    run_dir = _run_dir("abc123")
    _write_report(
        run_dir,
        {
            "passed": True,
            "run_id": "abc123",
            "artifacts_dir": str(run_dir),
            "failed_stage": None,
            "failure_reason": None,
            "report": {
                "run_id": "abc123",
                "spec_name": "vector_add_fp32",
                "stages_executed": ["interview", "codegen", "correctness", "performance", "polish"],
                "stage_traces": [
                    {
                        "stage_name": "interview",
                        "attempts": 1,
                        "succeeded": True,
                        "model_used": "claude-opus-4-7",
                        "tokens_in": 10,
                        "tokens_out": 20,
                        "cache_read_tokens": 0,
                        "latency_seconds": 1.2,
                    },
                    {
                        "stage_name": "codegen",
                        "attempts": 2,
                        "succeeded": True,
                        "model_used": "claude-sonnet-4-6",
                        "tokens_in": 30,
                        "tokens_out": 40,
                        "cache_read_tokens": 5,
                        "latency_seconds": 2.3,
                    },
                    {
                        "stage_name": "correctness",
                        "attempts": 1,
                        "succeeded": True,
                        "model_used": "none",
                        "tokens_in": 0,
                        "tokens_out": 0,
                        "cache_read_tokens": 0,
                        "latency_seconds": 0.4,
                    }
                ],
                "total_llm_tokens_in": 40,
                "total_llm_tokens_out": 60,
                "total_cost_usd": 0.0,
                "wall_time_seconds": 3.4,
                "warnings": [],
            },
            "correctness": {
                "passed": True,
                "max_abs_err": 0.0,
                "max_rel_err": 0.0,
                "shapes_tested": [[128]],
                "failing_inputs": [],
            },
            "performance": {
                "speedup_vs_reference": 1.5,
                "speedup_vs_torch_compile": 1.1,
                "achieved_gbps": 512.25,
                "below_target": False,
                "notes": ["benchmark stable"],
            },
        },
    )
    benchmark_path = run_dir / "stage4_performance" / "benchmark.json"
    benchmark_path.parent.mkdir()
    benchmark_path.write_text("{}", encoding="utf-8")
    polish_status_path = run_dir / "stage5_polish" / "status.json"
    polished_kernel_path = run_dir / "stage5_polish" / "final" / "kernel.cu"
    polished_kernel_path.parent.mkdir(parents=True)
    polish_status_path.write_text(
        json.dumps({"accepted": True, "reason": "validated", "kernel_cu_path": str(polished_kernel_path)}),
        encoding="utf-8",
    )
    polished_kernel_path.write_text("polished", encoding="utf-8")
    repair_report_path = run_dir / "stage3_repair" / "attempt_01" / "correctness_report.json"
    repair_kernel_path = run_dir / "stage3_repair" / "attempt_01" / "codegen" / "final" / "kernel.cu"
    repair_kernel_path.parent.mkdir(parents=True)
    repair_report_path.write_text("{}", encoding="utf-8")
    repair_kernel_path.write_text("fixed", encoding="utf-8")

    result = CliRunner().invoke(app, ["show-report", str(run_dir)])

    assert result.exit_code == 0
    assert "Run: abc123" in result.stdout
    assert "Status: PASS" in result.stdout
    assert "Spec: vector_add_fp32" in result.stdout
    assert "Stages: interview -> codegen -> correctness -> performance -> polish" in result.stdout
    assert "LLM tokens: 40 in / 60 out" in result.stdout
    assert "Stage traces:" in result.stdout
    assert "- interview: ok attempts=1 model=claude-opus-4-7 tokens=10/20 cache_read=0" in result.stdout
    assert "- codegen: ok attempts=2 model=claude-sonnet-4-6 tokens=30/40 cache_read=5" in result.stdout
    assert "- correctness: ok attempts=1 model=none tokens=0/0 cache_read=0" in result.stdout
    assert "Correctness: PASS" in result.stdout
    assert "Performance: speedup_vs_reference=1.50, speedup_vs_torch_compile=1.10" in result.stdout
    assert "Bandwidth: achieved_gbps=512.25" in result.stdout
    assert "Below target: false" in result.stdout
    assert "Performance notes: benchmark stable" in result.stdout
    assert f"Benchmark: {benchmark_path}" in result.stdout
    assert "Polish: accepted" in result.stdout
    assert f"Polished kernel: {polished_kernel_path}" in result.stdout
    assert "Correctness repairs: 1" in result.stdout
    assert f"- correctness_report: {repair_report_path}" in result.stdout
    assert f"- repaired_kernel: {repair_kernel_path}" in result.stdout
    assert f"Artifacts: {run_dir}" in result.stdout


def test_show_report_prints_failure_summary() -> None:
    run_dir = _run_dir("failed")
    _write_report(
        run_dir,
        {
            "passed": False,
            "run_id": "failed",
            "artifacts_dir": str(run_dir),
            "failed_stage": 3,
            "failure_reason": "correctness check failed",
            "report": {
                "run_id": "failed",
                "spec_name": "identity",
                "stages_executed": ["interview", "codegen", "correctness"],
                "stage_traces": [
                    {
                        "stage_name": "correctness",
                        "attempts": 1,
                        "succeeded": False,
                        "model_used": "none",
                    }
                ],
                "total_llm_tokens_in": 5,
                "total_llm_tokens_out": 6,
                "wall_time_seconds": 7.8,
                "warnings": ["below perf target"],
            },
            "correctness": {
                "passed": False,
                "max_abs_err": float("inf"),
                "max_rel_err": float("inf"),
                "shapes_tested": [[128]],
                "failing_inputs": [{"shape": [128], "error": "backend mismatch"}],
            },
            "performance": None,
        },
    )

    result = CliRunner().invoke(app, ["show-report", str(run_dir)])

    assert result.exit_code == 0
    assert "Status: FAIL" in result.stdout
    assert "Failed stage: 3" in result.stdout
    assert "Reason: correctness check failed" in result.stdout
    assert "Stage traces:" in result.stdout
    assert "- correctness: failed attempts=1 model=none tokens=0/0 cache_read=0" in result.stdout
    assert "Correctness: FAIL max_abs_err=inf max_rel_err=inf" in result.stdout
    assert "First failure: shape=[128] error=backend mismatch" in result.stdout
    assert "Warnings: below perf target" in result.stdout


def test_show_report_fails_when_report_is_missing() -> None:
    run_dir = _run_dir("missing")
    result = CliRunner().invoke(app, ["show-report", str(run_dir)])

    assert result.exit_code == 1
    assert "report.json not found" in result.stdout


def test_latest_report_prints_newest_report_summary() -> None:
    runs_root = _run_dir("runs-root")
    older = runs_root / "older"
    newer = runs_root / "newer"
    older.mkdir()
    newer.mkdir()
    _write_report(older, _minimal_report(run_id="older", run_dir=older, spec_name="old_spec"))
    _write_report(newer, _minimal_report(run_id="newer", run_dir=newer, spec_name="new_spec"))
    os.utime(older / "report.json", (1_700_000_000, 1_700_000_000))
    os.utime(newer / "report.json", (1_800_000_000, 1_800_000_000))

    result = CliRunner().invoke(app, ["latest-report", str(runs_root)])

    assert result.exit_code == 0
    assert "Run: newer" in result.stdout
    assert "Spec: new_spec" in result.stdout
    assert f"Artifacts: {newer}" in result.stdout
    assert "old_spec" not in result.stdout


def test_latest_report_fails_when_no_reports_exist() -> None:
    runs_root = _run_dir("empty-runs")

    result = CliRunner().invoke(app, ["latest-report", str(runs_root)])

    assert result.exit_code == 1
    assert "no report.json files found" in result.stdout


def test_eval_command_runs_suite_and_prints_summary(tmp_path: Path, monkeypatch: object) -> None:
    from _pytest.monkeypatch import MonkeyPatch

    typed_monkeypatch = monkeypatch
    assert isinstance(typed_monkeypatch, MonkeyPatch)
    suite_root = tmp_path / "internal"
    out_dir = tmp_path / "results"
    calls: list[dict[str, object]] = []

    def fake_run_eval_suite(**kwargs: object) -> EvalSummary:
        calls.append(kwargs)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "results.csv"
        markdown_path = out_dir / "summary.md"
        csv_path.write_text("kernel,passed\nvector_add,true\n", encoding="utf-8")
        markdown_path.write_text("# summary\n", encoding="utf-8")
        return EvalSummary(
            rows=[
                EvalRow(
                    kernel="vector_add",
                    passed=True,
                    run_id="abc123",
                    failed_stage=None,
                    failure_reason="",
                    speedup_vs_torch_compile=1.2,
                    speedup_vs_reference=1.3,
                    below_target=False,
                    artifacts_dir="/tmp/run",
                )
            ],
            out_dir=out_dir,
            csv_path=csv_path,
            markdown_path=markdown_path,
        )

    typed_monkeypatch.setattr("evals.runner.run_eval_suite", fake_run_eval_suite)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "--suite",
            str(suite_root),
            "--out",
            str(out_dir),
            "--target",
            "sm_90",
            "--only",
            "vector_add,clamp",
            "--limit",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert calls
    assert calls[0]["suite_root"] == suite_root
    assert calls[0]["out_dir"] == out_dir
    assert calls[0]["target"] == "sm_90"
    assert calls[0]["only"] == {"vector_add", "clamp"}
    assert calls[0]["limit"] == 2
    assert calls[0]["progress"] is not None
    assert "Eval complete: 1/1 passed" in result.stdout
    assert f"CSV: {out_dir / 'results.csv'}" in result.stdout
    assert f"Summary: {out_dir / 'summary.md'}" in result.stdout


def _write_report(run_dir: Path, payload: dict[str, object]) -> None:
    (run_dir / "report.json").write_text(json.dumps(payload), encoding="utf-8")


def _minimal_report(*, run_id: str, run_dir: Path, spec_name: str) -> dict[str, object]:
    return {
        "passed": True,
        "run_id": run_id,
        "artifacts_dir": str(run_dir),
        "failed_stage": None,
        "failure_reason": None,
        "report": {
            "run_id": run_id,
            "spec_name": spec_name,
            "stages_executed": ["interview"],
            "stage_traces": [],
            "total_llm_tokens_in": 1,
            "total_llm_tokens_out": 2,
            "warnings": [],
        },
        "correctness": {
            "passed": True,
            "max_abs_err": 0.0,
            "max_rel_err": 0.0,
            "shapes_tested": [[128]],
            "failing_inputs": [],
        },
        "performance": None,
    }


def _run_dir(name: str) -> Path:
    run_dir = Path(".test_artifacts") / "cli-tests" / f"{name}-{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True)
    return run_dir
