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

    if not payload.get("passed"):
        typer.echo(f"Failed stage: {payload.get('failed_stage')}")
        typer.echo(f"Reason: {payload.get('failure_reason')}")

    if isinstance(correctness, dict):
        _print_correctness(correctness)
    else:
        typer.echo("Correctness: not available")

    if isinstance(performance, dict):
        typer.echo(
            "Performance: "
            f"speedup_vs_reference={float(performance.get('speedup_vs_reference', 0.0)):.2f}, "
            f"speedup_vs_torch_compile={float(performance.get('speedup_vs_torch_compile', 0.0)):.2f}"
        )
    else:
        typer.echo("Performance: not available")

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
