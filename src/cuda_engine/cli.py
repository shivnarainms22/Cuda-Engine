import json
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(help="CUDA synthesis engine CLI.")


@app.callback()
def main() -> None:
    """Command-line entry point placeholder for M0."""


@app.command("show-report")
def show_report(run_dir: Path) -> None:
    """Print a compact summary for a run directory containing report.json."""
    report_path = run_dir / "report.json"
    if not report_path.exists():
        typer.echo(f"report.json not found: {report_path}")
        raise typer.Exit(code=1)
    _print_report_summary(report_path)


@app.command("latest-report")
def latest_report(runs_root: Path) -> None:
    """Print the newest report.json summary under a runs root."""
    report_path = _latest_report_path(runs_root)
    if report_path is None:
        typer.echo(f"no report.json files found under: {runs_root}")
        raise typer.Exit(code=1)
    _print_report_summary(report_path)


def _print_report_summary(report_path: Path) -> None:
    payload = _load_report(report_path)
    report = _dict(payload.get("report"))
    correctness = payload.get("correctness")
    performance = payload.get("performance")

    typer.echo(f"Run: {payload.get('run_id', report.get('run_id', 'unknown'))}")
    typer.echo(f"Status: {'PASS' if payload.get('passed') else 'FAIL'}")
    typer.echo(f"Spec: {report.get('spec_name', 'unknown')}")
    typer.echo(f"Stages: {' -> '.join(_strings(report.get('stages_executed')))}")
    typer.echo(
        "LLM tokens: "
        f"{int(report.get('total_llm_tokens_in', 0))} in / "
        f"{int(report.get('total_llm_tokens_out', 0))} out"
    )
    _print_stage_traces(report.get("stage_traces"))

    if not payload.get("passed"):
        typer.echo(f"Failed stage: {payload.get('failed_stage')}")
        typer.echo(f"Reason: {payload.get('failure_reason')}")

    if isinstance(correctness, dict):
        _print_correctness(correctness)
    else:
        typer.echo("Correctness: not available")

    _print_performance(performance, report_path.parent)
    _print_polish_artifacts(report_path.parent)
    _print_repair_artifacts(report_path.parent)

    warnings = _strings(report.get("warnings"))
    if warnings:
        typer.echo(f"Warnings: {', '.join(warnings)}")

    typer.echo(f"Artifacts: {payload.get('artifacts_dir', str(report_path.parent))}")


def _latest_report_path(runs_root: Path) -> Path | None:
    if not runs_root.exists():
        return None
    report_paths = [path for path in runs_root.rglob("report.json") if path.is_file()]
    if not report_paths:
        return None
    return max(report_paths, key=lambda path: path.stat().st_mtime)


def _load_report(report_path: Path) -> dict[str, Any]:
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        typer.echo(f"report.json could not be decoded: {exc}")
        raise typer.Exit(code=1) from exc
    if not isinstance(data, dict):
        typer.echo("report.json must contain an object")
        raise typer.Exit(code=1)
    return data


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _print_stage_traces(value: object) -> None:
    if not isinstance(value, list) or not value:
        return

    typer.echo("Stage traces:")
    for item in value:
        trace = _dict(item)
        status = "ok" if trace.get("succeeded") else "failed"
        typer.echo(
            f"- {trace.get('stage_name', 'unknown')}: "
            f"{status} "
            f"attempts={int(trace.get('attempts', 0))} "
            f"model={trace.get('model_used', 'unknown')} "
            f"tokens={int(trace.get('tokens_in', 0))}/{int(trace.get('tokens_out', 0))} "
            f"cache_read={int(trace.get('cache_read_tokens', 0))}"
        )


def _print_correctness(correctness: dict[str, Any]) -> None:
    if correctness.get("passed"):
        typer.echo("Correctness: PASS")
        return

    typer.echo(
        "Correctness: FAIL "
        f"max_abs_err={correctness.get('max_abs_err')} "
        f"max_rel_err={correctness.get('max_rel_err')}"
    )
    failing_inputs = correctness.get("failing_inputs")
    if isinstance(failing_inputs, list) and failing_inputs:
        first_failure = _dict(failing_inputs[0])
        typer.echo(
            "First failure: "
            f"shape={first_failure.get('shape')} "
            f"error={first_failure.get('error')}"
        )


def _print_performance(performance: object, run_dir: Path) -> None:
    if not isinstance(performance, dict):
        typer.echo("Performance: not available")
        return

    typer.echo(
        "Performance: "
        f"speedup_vs_reference={float(performance.get('speedup_vs_reference', 0.0)):.2f}, "
        f"speedup_vs_torch_compile={float(performance.get('speedup_vs_torch_compile', 0.0)):.2f}"
    )

    achieved_gbps = performance.get("achieved_gbps")
    if achieved_gbps is not None:
        typer.echo(f"Bandwidth: achieved_gbps={float(achieved_gbps):.2f}")

    typer.echo(f"Below target: {str(bool(performance.get('below_target', False))).lower()}")

    notes = _strings(performance.get("notes"))
    if notes:
        typer.echo(f"Performance notes: {', '.join(notes)}")

    benchmark_path = run_dir / "stage4_performance" / "benchmark.json"
    if benchmark_path.exists():
        typer.echo(f"Benchmark: {benchmark_path}")


def _print_repair_artifacts(run_dir: Path) -> None:
    repair_root = run_dir / "stage3_repair"
    if not repair_root.exists():
        return

    attempt_dirs = sorted(path for path in repair_root.glob("attempt_*") if path.is_dir())
    if not attempt_dirs:
        return

    typer.echo(f"Correctness repairs: {len(attempt_dirs)}")
    for attempt_dir in attempt_dirs:
        report_path = attempt_dir / "correctness_report.json"
        kernel_path = attempt_dir / "codegen" / "final" / "kernel.cu"
        if report_path.exists():
            typer.echo(f"- correctness_report: {report_path}")
        if kernel_path.exists():
            typer.echo(f"- repaired_kernel: {kernel_path}")


def _print_polish_artifacts(run_dir: Path) -> None:
    status_path = run_dir / "stage5_polish" / "status.json"
    if not status_path.exists():
        return

    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        typer.echo(f"Polish: status unreadable at {status_path}")
        return
    if not isinstance(status, dict):
        typer.echo(f"Polish: status unreadable at {status_path}")
        return

    accepted = bool(status.get("accepted", False))
    typer.echo(f"Polish: {'accepted' if accepted else 'rejected'}")
    reason = status.get("reason")
    if reason:
        typer.echo(f"Polish reason: {reason}")

    kernel_path = run_dir / "stage5_polish" / "final" / "kernel.cu"
    if not kernel_path.exists():
        raw_path = status.get("kernel_cu_path")
        kernel_path = Path(str(raw_path)) if raw_path else kernel_path
    if kernel_path.exists() or status.get("kernel_cu_path"):
        typer.echo(f"Polished kernel: {kernel_path}")
