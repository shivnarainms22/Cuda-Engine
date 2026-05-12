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
    _kernel_dir(
        suite_root,
        "vector_add",
        prompt="Add vectors.",
        reference_body="return x",
        shapes=[(128,), (1024,), (4097,)],
    )
    incomplete = suite_root / "missing_reference"
    incomplete.mkdir(parents=True)
    (incomplete / "prompt.txt").write_text("missing reference", encoding="utf-8")

    kernels = discover_kernels(suite_root)

    assert [kernel.name for kernel in kernels] == ["vector_add"]
    assert kernels[0].prompt == "Add vectors."
    assert kernels[0].correctness_shapes == ((128,), (1024,), (4097,))
    assert kernels[0].reference(3) == 3


def test_discover_kernels_requires_shapes_and_notes(tmp_path: Path) -> None:
    from evals.runner import discover_kernels

    suite_root = tmp_path / "internal"
    missing_shapes = _kernel_dir(
        suite_root,
        "missing_shapes",
        prompt="Missing shapes.",
        reference_body="return x",
        shapes=[(128,), (1024,), (4097,)],
    )
    (missing_shapes / "shapes.yaml").unlink()
    missing_notes = _kernel_dir(
        suite_root,
        "missing_notes",
        prompt="Missing notes.",
        reference_body="return x",
        shapes=[(128,), (1024,), (4097,)],
    )
    (missing_notes / "notes.md").unlink()
    _kernel_dir(
        suite_root,
        "complete",
        prompt="Complete.",
        reference_body="return x",
        shapes=[(128,), (1024,), (4097,)],
    )

    kernels = discover_kernels(suite_root)

    assert [kernel.name for kernel in kernels] == ["complete"]


def test_run_eval_suite_passes_kernel_shapes_to_config(tmp_path: Path) -> None:
    from evals.runner import run_eval_suite

    suite_root = tmp_path / "internal"
    _kernel_dir(
        suite_root,
        "matrix_kernel",
        prompt="Matrix kernel.",
        reference_body="return x",
        shapes=[(16, 64), (32, 128), (4, 1024)],
    )
    seen_shapes: list[tuple[tuple[int, ...], ...]] = []

    def fake_synthesize(
        prompt: str,
        reference: object,
        target: str,
        config: SynthesisConfig,
    ) -> SynthesisResult:
        seen_shapes.append(config.correctness_shapes)
        return _result(
            kernel_name="matrix_kernel",
            passed=True,
            run_id="matrix",
            artifacts_dir="/tmp/matrix",
            speedup=1.0,
        )

    run_eval_suite(
        suite_root=suite_root,
        out_dir=tmp_path / "results",
        synthesize_fn=fake_synthesize,
    )

    assert seen_shapes == [((16, 64), (32, 128), (4, 1024))]


def test_run_eval_suite_writes_csv_markdown_and_per_kernel_json(tmp_path: Path) -> None:
    from evals.runner import run_eval_suite

    suite_root = tmp_path / "internal"
    _kernel_dir(
        suite_root,
        "fast_kernel",
        prompt="Fast kernel.",
        reference_body="return x + 1",
        shapes=[(128,), (1024,), (4097,)],
    )
    _kernel_dir(
        suite_root,
        "failed_kernel",
        prompt="Fail kernel.",
        reference_body="return x",
        shapes=[(128,), (1024,), (4097,)],
    )
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
        "failure_kind",
        "speedup_vs_torch_compile",
        "speedup_vs_reference",
        "below_target",
        "baseline_status",
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


def test_run_eval_suite_classifies_external_api_failures(tmp_path: Path) -> None:
    from evals.runner import run_eval_suite

    suite_root = tmp_path / "internal"
    _kernel_dir(
        suite_root,
        "credit_blocked",
        prompt="Credit blocked.",
        reference_body="return x",
        shapes=[(128,), (1024,), (4097,)],
    )
    _kernel_dir(
        suite_root,
        "stage_failed",
        prompt="Stage failed.",
        reference_body="return x",
        shapes=[(128,), (1024,), (4097,)],
    )

    def fake_synthesize(
        prompt: str,
        reference: object,
        target: str,
        config: SynthesisConfig,
    ) -> SynthesisResult:
        if prompt == "Credit blocked.":
            raise RuntimeError(
                "BadRequestError: Error code: 400 - credit balance is too low "
                "to access the Anthropic API"
            )
        return _result(
            kernel_name="stage_failed",
            passed=False,
            run_id="stage-failed",
            artifacts_dir="/tmp/stage-failed",
            speedup=None,
            failed_stage=3,
            failure_reason="correctness check failed",
        )

    run_eval_suite(
        suite_root=suite_root,
        out_dir=tmp_path / "results",
        synthesize_fn=fake_synthesize,
    )

    csv_rows = {row["kernel"]: row for row in _csv_rows(tmp_path / "results" / "results.csv")}
    assert csv_rows["credit_blocked"]["failure_kind"] == "external_error"
    assert csv_rows["stage_failed"]["failure_kind"] == "stage_failure"

    external_json = json.loads(
        (tmp_path / "results" / "kernels" / "credit_blocked.json").read_text(encoding="utf-8")
    )
    assert external_json["failure_kind"] == "external_error"

    summary_md = (tmp_path / "results" / "summary.md").read_text(encoding="utf-8")
    assert "- External/API failures: 1" in summary_md
    assert "- Stage/kernel failures: 1" in summary_md
    assert "| credit_blocked | FAIL |  |  | external_error |" in summary_md


def test_run_eval_suite_summary_reports_m3_metrics(tmp_path: Path) -> None:
    from evals.runner import run_eval_suite

    suite_root = tmp_path / "internal"
    for name in ("fast_a", "fast_b", "slow_a", "slow_b", "failed"):
        _kernel_dir(
            suite_root,
            name,
            prompt=f"{name}.",
            reference_body="return x",
            shapes=[(128,), (1024,), (4097,)],
        )

    speedups = {
        "fast_a.": 1.30,
        "fast_b.": 1.10,
        "slow_a.": 0.90,
        "slow_b.": 0.70,
    }

    def fake_synthesize(
        prompt: str,
        reference: object,
        target: str,
        config: SynthesisConfig,
    ) -> SynthesisResult:
        if prompt == "failed.":
            return _result(
                kernel_name="failed",
                passed=False,
                run_id="failed",
                artifacts_dir="/tmp/failed",
                speedup=None,
                failed_stage=3,
                failure_reason="correctness failed",
            )
        return _result(
            kernel_name=prompt.removesuffix("."),
            passed=True,
            run_id=prompt.removesuffix("."),
            artifacts_dir=f"/tmp/{prompt.removesuffix('.')}",
            speedup=speedups[prompt],
            below_target=speedups[prompt] < 1.0,
        )

    run_eval_suite(
        suite_root=suite_root,
        out_dir=tmp_path / "results",
        synthesize_fn=fake_synthesize,
    )

    summary_md = (tmp_path / "results" / "summary.md").read_text(encoding="utf-8")
    assert "## M3 Metrics" in summary_md
    assert "- Pass rate: 4/5 (80.0%)" in summary_md
    assert "- Median speedup vs torch.compile: 1.00x" in summary_md
    assert "- P25 speedup vs torch.compile: 0.85x" in summary_md
    assert "- fast_1 kernels (>1.0x with measured baseline): 2/5" in summary_md
    assert "- Below target kernels (with measured baseline): 2/5" in summary_md
    assert "- baseline_failed (not counted in fast_1): 0/5" in summary_md


def test_run_eval_suite_marks_baseline_regressions(tmp_path: Path) -> None:
    from evals.runner import run_eval_suite

    suite_root = tmp_path / "internal"
    _kernel_dir(
        suite_root,
        "became_slow",
        prompt="Slow now.",
        reference_body="return x",
        shapes=[(128,), (1024,), (4097,)],
    )
    _kernel_dir(
        suite_root,
        "became_fail",
        prompt="Fail now.",
        reference_body="return x",
        shapes=[(128,), (1024,), (4097,)],
    )
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "results.csv").write_text(
        "kernel,passed,run_id,failed_stage,failure_reason,failure_kind,"
        "speedup_vs_torch_compile,speedup_vs_reference,below_target,artifacts_dir,regression\n"
        "became_slow,true,old,,,,1.50,1.50,false,/old,\n"
        "became_fail,true,old,,,,1.20,1.20,false,/old,\n",
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


def test_run_eval_suite_reports_progress_and_skips_completed_kernels(tmp_path: Path) -> None:
    from evals.runner import run_eval_suite

    suite_root = tmp_path / "internal"
    _kernel_dir(
        suite_root,
        "already_done",
        prompt="Done.",
        reference_body="return x",
        shapes=[(128,), (1024,), (4097,)],
    )
    _kernel_dir(
        suite_root,
        "new_kernel",
        prompt="New.",
        reference_body="return x",
        shapes=[(128,), (1024,), (4097,)],
    )
    out_dir = tmp_path / "results"
    completed_dir = out_dir / "kernels"
    completed_dir.mkdir(parents=True)
    (completed_dir / "already_done.json").write_text(
        json.dumps(
            {
                "kernel": "already_done",
                "passed": True,
                "run_id": "old",
                "failed_stage": None,
                "failure_reason": "",
                "failure_kind": "",
                "speedup_vs_torch_compile": 1.1,
                "speedup_vs_reference": 1.2,
                "below_target": False,
                "artifacts_dir": "/old",
                "regression": "",
            }
        ),
        encoding="utf-8",
    )
    progress: list[str] = []
    calls: list[str] = []

    def fake_synthesize(
        prompt: str,
        reference: object,
        target: str,
        config: SynthesisConfig,
    ) -> SynthesisResult:
        calls.append(prompt)
        return _result(
            kernel_name="new_kernel",
            passed=True,
            run_id="new",
            artifacts_dir="/new",
            speedup=1.0,
        )

    summary = run_eval_suite(
        suite_root=suite_root,
        out_dir=out_dir,
        limit=1,
        progress=progress.append,
        synthesize_fn=fake_synthesize,
    )

    assert [row.kernel for row in summary.rows] == ["already_done", "new_kernel"]
    assert calls == ["New."]
    assert progress == [
        "[1/2] SKIP already_done (existing result)",
        "[2/2] RUN new_kernel",
        "[2/2] DONE new_kernel passed=True speedup=1.00",
    ]


def test_run_eval_suite_only_filters_kernel_names(tmp_path: Path) -> None:
    from evals.runner import run_eval_suite

    suite_root = tmp_path / "internal"
    _kernel_dir(
        suite_root,
        "first",
        prompt="First.",
        reference_body="return x",
        shapes=[(128,), (1024,), (4097,)],
    )
    _kernel_dir(
        suite_root,
        "second",
        prompt="Second.",
        reference_body="return x",
        shapes=[(128,), (1024,), (4097,)],
    )
    calls: list[str] = []

    def fake_synthesize(
        prompt: str,
        reference: object,
        target: str,
        config: SynthesisConfig,
    ) -> SynthesisResult:
        calls.append(prompt)
        return _result(
            kernel_name="second",
            passed=True,
            run_id="second",
            artifacts_dir="/second",
            speedup=1.0,
        )

    summary = run_eval_suite(
        suite_root=suite_root,
        out_dir=tmp_path / "results",
        only={"second"},
        synthesize_fn=fake_synthesize,
    )

    assert [row.kernel for row in summary.rows] == ["second"]
    assert calls == ["Second."]


def _kernel_dir(
    suite_root: Path,
    name: str,
    *,
    prompt: str,
    reference_body: str,
    shapes: list[tuple[int, ...]],
) -> Path:
    kernel_dir = suite_root / name
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (kernel_dir / "reference.py").write_text(
        f"def reference(x):\n    {reference_body}\n",
        encoding="utf-8",
    )
    shapes_text = "\n".join(
        f"- [{', '.join(str(dim) for dim in shape)}]" for shape in shapes
    )
    (kernel_dir / "shapes.yaml").write_text(f"{shapes_text}\n", encoding="utf-8")
    (kernel_dir / "notes.md").write_text("test fixture\n", encoding="utf-8")
    return kernel_dir


def _result(
    *,
    kernel_name: str,
    passed: bool,
    run_id: str,
    artifacts_dir: str,
    speedup: float | None,
    below_target: bool = False,
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
            below_target=below_target,
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
