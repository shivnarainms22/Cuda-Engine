import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any

import typer

from cuda_engine.config import SynthesisConfig

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


@app.command("synthesize")
def synthesize_cmd(
    prompt: Annotated[
        str | None,
        typer.Option(
            "--prompt",
            help="Prompt text. Mutually exclusive with --prompt-file.",
        ),
    ] = None,
    prompt_file: Annotated[
        Path | None,
        typer.Option("--prompt-file", help="Path to a text file containing the prompt."),
    ] = None,
    reference: Annotated[
        Path,
        typer.Option(
            "--reference",
            help="Python file defining the reference function (variable REFERENCE or reference()).",
        ),
    ] = ...,  # type: ignore[assignment]
    target: Annotated[str, typer.Option(help="CUDA target architecture.")] = "sm_80",
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Artifact root for the run. Defaults to ~/.cache/cuda_engine/runs/.",
        ),
    ] = None,
) -> None:
    """Synthesize a single CUDA kernel from a prompt + reference function."""
    if prompt is None and prompt_file is None:
        typer.echo("error: one of --prompt or --prompt-file is required")
        raise typer.Exit(code=2)
    if prompt is not None and prompt_file is not None:
        typer.echo("error: --prompt and --prompt-file are mutually exclusive")
        raise typer.Exit(code=2)

    if prompt is not None:
        prompt_text = prompt
    else:
        assert prompt_file is not None  # narrowed by validation above
        prompt_text = _read_text(prompt_file)
    reference_fn = _load_reference_from_path(reference)

    synthesize_fn = _resolve_synthesize_fn()
    config = SynthesisConfig(artifact_root=str(out)) if out is not None else SynthesisConfig()
    result = synthesize_fn(
        prompt=prompt_text,
        reference=reference_fn,
        target=target,
        config=config,
    )

    typer.echo(f"Run: {result.run_id}")
    typer.echo(f"Status: {'PASS' if result.passed else 'FAIL'}")
    typer.echo(f"Artifacts: {result.artifacts_dir}")
    if not result.passed:
        typer.echo(f"Failed stage: {result.failed_stage}")
        typer.echo(f"Reason: {result.failure_reason}")
        raise typer.Exit(code=1)


@app.command("inspect")
def inspect_run(
    run: Annotated[
        str,
        typer.Argument(
            help="Run id or path to a run directory containing report.json.",
        ),
    ],
    runs_root: Annotated[
        Path | None,
        typer.Option(
            "--runs-root",
            help="Directory containing run subdirectories. Defaults to ~/.cache/cuda_engine/runs.",
        ),
    ] = None,
) -> None:
    """Pretty-print the report for a synthesis run."""
    run_dir = _resolve_run_dir(run, runs_root)
    if run_dir is None:
        typer.echo(f"run not found: {run}")
        raise typer.Exit(code=1)
    report_path = run_dir / "report.json"
    if not report_path.exists():
        typer.echo(f"report.json not found: {report_path}")
        raise typer.Exit(code=1)
    _print_report_summary(report_path)


def _resolve_run_dir(run: str, runs_root: Path | None) -> Path | None:
    direct = Path(run)
    if direct.is_dir() and (direct / "report.json").exists():
        return direct
    root = runs_root if runs_root is not None else Path.home() / ".cache" / "cuda_engine" / "runs"
    if not root.exists():
        return None
    candidate = root / run
    if candidate.is_dir() and (candidate / "report.json").exists():
        return candidate
    # Tolerate truncated run_ids: pick the unique match if any.
    matches = [
        path for path in root.iterdir()
        if path.is_dir() and path.name.startswith(run) and (path / "report.json").exists()
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        typer.echo(f"could not read prompt file: {exc}")
        raise typer.Exit(code=1) from exc


def _load_reference_from_path(reference_path: Path) -> Any:
    if not reference_path.exists():
        typer.echo(f"reference file not found: {reference_path}")
        raise typer.Exit(code=1)
    module_name = f"cuda_engine_cli_reference_{reference_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, reference_path)
    if spec is None or spec.loader is None:
        typer.echo(f"could not load reference module: {reference_path}")
        raise typer.Exit(code=1)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        typer.echo(f"reference module failed to import: {exc}")
        raise typer.Exit(code=1) from exc
    reference = getattr(module, "REFERENCE", None) or getattr(module, "reference", None)
    if not callable(reference):
        typer.echo(f"reference file must define REFERENCE or reference(): {reference_path}")
        raise typer.Exit(code=1)
    return reference


def _resolve_synthesize_fn() -> Any:
    from cuda_engine import synthesize as synthesize_fn

    return synthesize_fn


@app.command("eval")
def eval_suite(
    out: Annotated[Path, typer.Option("--out", help="Directory for aggregate eval outputs.")],
    suite: Annotated[
        str,
        typer.Option(help="Suite name ('internal') or path to a suite directory."),
    ] = "internal",
    baseline: Annotated[
        Path | None,
        typer.Option(help="Optional prior results directory."),
    ] = None,
    target: Annotated[str, typer.Option(help="CUDA target architecture.")] = "sm_80",
    only: Annotated[
        str | None,
        typer.Option(help="Comma-separated kernel names to run."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(help="Maximum number of selected kernels to run."),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Skip kernels with existing JSON results."),
    ] = True,
) -> None:
    """Run an eval suite and write aggregate markdown/CSV results."""
    eval_runner = _load_eval_runner()
    suite_root = Path("evals") / "internal" if suite == "internal" else Path(suite)
    summary = eval_runner.run_eval_suite(
        suite_root=suite_root,
        out_dir=out,
        baseline_dir=baseline,
        target=target,
        config=SynthesisConfig(),
        only=_parse_only(only),
        limit=limit,
        resume=resume,
        progress=typer.echo,
    )
    passed = sum(1 for row in summary.rows if row.passed)
    typer.echo(f"Eval complete: {passed}/{len(summary.rows)} passed")
    typer.echo(f"CSV: {summary.csv_path}")
    typer.echo(f"Summary: {summary.markdown_path}")


def _load_eval_runner() -> ModuleType:
    try:
        from evals import runner as eval_runner

        return eval_runner
    except ModuleNotFoundError as exc:
        for root in (Path.cwd(), *Path.cwd().parents):
            runner_path = root / "evals" / "runner.py"
            if runner_path.exists():
                spec = importlib.util.spec_from_file_location(
                    "cuda_engine_source_eval_runner",
                    runner_path,
                )
                if spec is None or spec.loader is None:
                    break
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                return module
        raise ModuleNotFoundError(
            "could not import evals.runner; run this command from a cuda-engine source checkout"
        ) from exc


def _parse_only(value: str | None) -> set[str] | None:
    if value is None:
        return None
    names = {item.strip() for item in value.split(",") if item.strip()}
    return names or None


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

    def _fmt(value: object) -> str:
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.2f}"  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "n/a"

    typer.echo(
        "Performance: "
        f"speedup_vs_reference={_fmt(performance.get('speedup_vs_reference'))}, "
        f"speedup_vs_torch_compile={_fmt(performance.get('speedup_vs_torch_compile'))}"
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
