import csv
import json
from pathlib import Path

from cuda_engine.config import SynthesisConfig
from cuda_engine.models import (
    CorrectnessReport,
    PerformanceReport,
    SynthesisReport,
    SynthesisResult,
)


def test_discover_kernels_returns_complete_kernel_dirs(tmp_path: Path) -> None:
    from evals.runner import discover_kernels

    suite_root = tmp_path / "internal"
    _kernel_dir(suite_root, "vector_add", prompt="Add vectors.", reference_body="return x")
    incomplete = suite_root / "missing_reference"
    incomplete.mkdir(parents=True)
    (incomplete / "prompt.txt").write_text("missing reference", encoding="utf-8")

    kernels = discover_kernels(suite_root)

    assert [kernel.name for kernel in kernels] == ["vector_add"]
    assert kernels[0].prompt == "Add vectors."
    assert kernels[0].reference(3) == 3


def test_run_eval_suite_writes_csv_markdown_and_per_kernel_json(tmp_path: Path) -> None:
    from evals.runner import run_eval_suite

    suite_root = tmp_path / "internal"
    _kernel_dir(suite_root, "fast_kernel", prompt="Fast kernel.", reference_body="return x + 1")
    _kernel_dir(suite_root, "failed_kernel", prompt="Fail kernel.", reference_body="return x")
    out_dir = tmp_path / "results"
    calls: list[tuple[str, str, str]] = []

    def fake_synthesize(
        prompt: str,
        reference: object,
        target: str,
        config: SynthesisConfig,
    ) -> SynthesisResult:
        assert callable(reference)
        calls.append((prompt, target, config.artifact_root or ""))
        if prompt == "Fast kernel.":
            return _result(
                kernel_name="fast_kernel",
                passed=True,
                run_id="run-fast",
                artifacts_dir=str(Path(config.artifact_root or "") / "run-fast"),
                speedup=1.25,
            )
        return _result(
            kernel_name="failed_kernel",
            passed=False,
            run_id="run-failed",
            artifacts_dir=str(Path(config.artifact_root or "") / "run-failed"),
            speedup=None,
            failed_stage=3,
            failure_reason="correctness check failed",
        )

    summary = run_eval_suite(
        suite_root=suite_root,
        out_dir=out_dir,
        target="sm_80",
        config=SynthesisConfig(),
        synthesize_fn=fake_synthesize,
    )

    assert [row.kernel for row in summary.rows] == ["failed_kernel", "fast_kernel"]
    assert calls == [
        ("Fail kernel.", "sm_80", str(out_dir / "artifacts" / "failed_kernel")),
        ("Fast kernel.", "sm_80", str(out_dir / "artifacts" / "fast_kernel")),
    ]

    csv_rows = _csv_rows(out_dir / "results.csv")
    assert list(csv_rows[0]) == [
        "kernel",
        "passed",
        "run_id",
        "failed_stage",
        "failure_reason",
        "speedup_vs_torch_compile",
        "speedup_vs_reference",
        "below_target",
        "artifacts_dir",
        "regression",
    ]
    assert csv_rows[0]["kernel"] == "failed_kernel"
    assert csv_rows[0]["passed"] == "false"
    assert csv_rows[0]["failed_stage"] == "3"
    assert csv_rows[1]["kernel"] == "fast_kernel"
    assert csv_rows[1]["speedup_vs_torch_compile"] == "1.25"

    summary_md = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert "# CUDA Engine Eval Summary" in summary_md
    assert "Pass rate: 1/2" in summary_md
    assert "| fast_kernel | PASS | 1.25 |" in summary_md

    per_kernel = json.loads((out_dir / "kernels" / "fast_kernel.json").read_text(encoding="utf-8"))
    assert per_kernel["kernel"] == "fast_kernel"
    assert per_kernel["passed"] is True


def test_run_eval_suite_marks_baseline_regressions(tmp_path: Path) -> None:
    from evals.runner import run_eval_suite

    suite_root = tmp_path / "internal"
    _kernel_dir(suite_root, "became_slow", prompt="Slow now.", reference_body="return x")
    _kernel_dir(suite_root, "became_fail", prompt="Fail now.", reference_body="return x")
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "results.csv").write_text(
        "kernel,passed,run_id,failed_stage,failure_reason,speedup_vs_torch_compile,"
        "speedup_vs_reference,below_target,artifacts_dir,regression\n"
        "became_slow,true,old,,,1.50,1.50,false,/old,\n"
        "became_fail,true,old,,,1.20,1.20,false,/old,\n",
        encoding="utf-8",
    )

    def fake_synthesize(
        prompt: str,
        reference: object,
        target: str,
        config: SynthesisConfig,
    ) -> SynthesisResult:
        if prompt == "Slow now.":
            return _result(
                kernel_name="became_slow",
                passed=True,
                run_id="slow",
                artifacts_dir="/tmp/slow",
                speedup=0.90,
            )
        return _result(
            kernel_name="became_fail",
            passed=False,
            run_id="fail",
            artifacts_dir="/tmp/fail",
            speedup=None,
            failed_stage=3,
            failure_reason="correctness check failed",
        )

    summary = run_eval_suite(
        suite_root=suite_root,
        out_dir=tmp_path / "results",
        baseline_dir=baseline,
        synthesize_fn=fake_synthesize,
    )

    regressions = {row.kernel: row.regression for row in summary.rows}
    assert regressions == {
        "became_fail": "pass_to_fail",
        "became_slow": "speedup_drop:1.50->0.90",
    }


def _kernel_dir(suite_root: Path, name: str, *, prompt: str, reference_body: str) -> Path:
    kernel_dir = suite_root / name
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (kernel_dir / "reference.py").write_text(
        f"def reference(x):\n    {reference_body}\n",
        encoding="utf-8",
    )
    (kernel_dir / "shapes.yaml").write_text("- [128]\n", encoding="utf-8")
    (kernel_dir / "notes.md").write_text("test fixture\n", encoding="utf-8")
    return kernel_dir


def _result(
    *,
    kernel_name: str,
    passed: bool,
    run_id: str,
    artifacts_dir: str,
    speedup: float | None,
    failed_stage: int | None = None,
    failure_reason: str | None = None,
) -> SynthesisResult:
    report = SynthesisReport(
        run_id=run_id,
        spec_name=kernel_name,
        stages_executed=["interview", "codegen", "correctness", "performance"],
    )
    correctness = CorrectnessReport(
        passed=passed,
        max_abs_err=0.0 if passed else 1.0,
        max_rel_err=0.0 if passed else 1.0,
        shapes_tested=[(128,)],
    )
    performance = (
        PerformanceReport(
            speedup_vs_reference=speedup,
            speedup_vs_torch_compile=speedup,
            below_target=False,
        )
        if speedup is not None
        else None
    )
    if passed:
        assert performance is not None
        return SynthesisResult.ok(
            run_id=run_id,
            artifacts_dir=artifacts_dir,
            report=report,
            correctness=correctness,
            performance=performance,
            kernel_callable=None,
        )
    return SynthesisResult.failed(
        stage=failed_stage or 0,
        reason=failure_reason or "failed",
        run_id=run_id,
        artifacts_dir=artifacts_dir,
        report=report,
        correctness=correctness,
    )


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
