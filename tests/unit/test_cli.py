import json
import uuid
from pathlib import Path

from typer.testing import CliRunner

from cuda_engine.cli import app


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
                    }
                ],
                "total_llm_tokens_in": 10,
                "total_llm_tokens_out": 20,
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
                "below_target": False,
            },
        },
    )

    result = CliRunner().invoke(app, ["show-report", str(run_dir)])

    assert result.exit_code == 0
    assert "Run: abc123" in result.stdout
    assert "Status: PASS" in result.stdout
    assert "Spec: vector_add_fp32" in result.stdout
    assert "Stages: interview -> codegen -> correctness -> performance -> polish" in result.stdout
    assert "LLM tokens: 10 in / 20 out" in result.stdout
    assert "Correctness: PASS" in result.stdout
    assert "Performance: speedup_vs_reference=1.50, speedup_vs_torch_compile=1.10" in result.stdout
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
    assert "Correctness: FAIL max_abs_err=inf max_rel_err=inf" in result.stdout
    assert "First failure: shape=[128] error=backend mismatch" in result.stdout
    assert "Warnings: below perf target" in result.stdout


def test_show_report_fails_when_report_is_missing() -> None:
    run_dir = _run_dir("missing")
    result = CliRunner().invoke(app, ["show-report", str(run_dir)])

    assert result.exit_code == 1
    assert "report.json not found" in result.stdout


def _write_report(run_dir: Path, payload: dict[str, object]) -> None:
    (run_dir / "report.json").write_text(json.dumps(payload), encoding="utf-8")


def _run_dir(name: str) -> Path:
    run_dir = Path(".test_artifacts") / "cli-tests" / f"{name}-{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True)
    return run_dir
