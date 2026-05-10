from __future__ import annotations

import csv
import importlib.util
import json
import traceback
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from cuda_engine import synthesize
from cuda_engine.config import SynthesisConfig
from cuda_engine.models import SynthesisResult

SynthFn = Callable[..., SynthesisResult]

CSV_COLUMNS = [
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


@dataclass(frozen=True)
class EvalKernel:
    name: str
    root: Path
    prompt: str
    reference: Callable[..., Any]


@dataclass(frozen=True)
class EvalRow:
    kernel: str
    passed: bool
    run_id: str
    failed_stage: int | None
    failure_reason: str
    speedup_vs_torch_compile: float | None
    speedup_vs_reference: float | None
    below_target: bool | None
    artifacts_dir: str
    regression: str = ""


@dataclass(frozen=True)
class EvalSummary:
    rows: list[EvalRow]
    out_dir: Path
    csv_path: Path
    markdown_path: Path


def discover_kernels(suite_root: Path) -> list[EvalKernel]:
    """Discover valid eval kernels under a source-checkout suite directory."""
    if not suite_root.exists():
        return []

    kernels: list[EvalKernel] = []
    for kernel_dir in sorted(path for path in suite_root.iterdir() if path.is_dir()):
        prompt_path = kernel_dir / "prompt.txt"
        reference_path = kernel_dir / "reference.py"
        if not prompt_path.exists() or not reference_path.exists():
            continue
        kernels.append(
            EvalKernel(
                name=kernel_dir.name,
                root=kernel_dir,
                prompt=prompt_path.read_text(encoding="utf-8").strip(),
                reference=_load_reference(reference_path),
            )
        )
    return kernels


def run_eval_suite(
    *,
    suite_root: Path,
    out_dir: Path,
    baseline_dir: Path | None = None,
    target: str = "sm_80",
    config: SynthesisConfig | None = None,
    synthesize_fn: SynthFn = synthesize,
) -> EvalSummary:
    out_dir.mkdir(parents=True, exist_ok=True)
    kernels_dir = out_dir / "kernels"
    kernels_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = out_dir / "artifacts"
    baseline = _load_baseline(baseline_dir)

    rows: list[EvalRow] = []
    for kernel in discover_kernels(suite_root):
        kernel_config = (config or SynthesisConfig()).model_copy(
            update={"artifact_root": str(artifact_root / kernel.name)}
        )
        row = _run_kernel(
            kernel=kernel,
            target=target,
            config=kernel_config,
            synthesize_fn=synthesize_fn,
        )
        row = _with_regression(row, baseline.get(row.kernel))
        rows.append(row)
        (kernels_dir / f"{kernel.name}.json").write_text(
            json.dumps(_row_to_json(row), indent=2),
            encoding="utf-8",
        )

    csv_path = out_dir / "results.csv"
    markdown_path = out_dir / "summary.md"
    _write_csv(csv_path, rows)
    _write_markdown(markdown_path, rows)
    return EvalSummary(rows=rows, out_dir=out_dir, csv_path=csv_path, markdown_path=markdown_path)


def _run_kernel(
    *,
    kernel: EvalKernel,
    target: str,
    config: SynthesisConfig,
    synthesize_fn: SynthFn,
) -> EvalRow:
    try:
        result = synthesize_fn(
            prompt=kernel.prompt,
            reference=kernel.reference,
            target=target,
            config=config,
        )
    except Exception as exc:  # pragma: no cover - exercised by real eval failures
        return EvalRow(
            kernel=kernel.name,
            passed=False,
            run_id="",
            failed_stage=None,
            failure_reason=f"{type(exc).__name__}: {exc}",
            speedup_vs_torch_compile=None,
            speedup_vs_reference=None,
            below_target=None,
            artifacts_dir=str(config.artifact_root or ""),
            regression="",
        )

    performance = result.performance
    return EvalRow(
        kernel=kernel.name,
        passed=result.passed,
        run_id=result.run_id,
        failed_stage=result.failed_stage,
        failure_reason=result.failure_reason or "",
        speedup_vs_torch_compile=(
            performance.speedup_vs_torch_compile if performance is not None else None
        ),
        speedup_vs_reference=(performance.speedup_vs_reference if performance is not None else None),
        below_target=(performance.below_target if performance is not None else None),
        artifacts_dir=result.artifacts_dir,
        regression="",
    )


def _load_reference(reference_path: Path) -> Callable[..., Any]:
    module_name = f"cuda_engine_eval_{reference_path.parent.name}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, reference_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load reference module: {reference_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ValueError(
            f"reference.py failed to import: {reference_path}\n{traceback.format_exc()}"
        ) from exc

    reference = getattr(module, "REFERENCE", None) or getattr(module, "reference", None)
    if not callable(reference):
        raise ValueError(f"reference.py must define REFERENCE or reference(): {reference_path}")
    return cast(Callable[..., Any], reference)


def _load_baseline(baseline_dir: Path | None) -> dict[str, dict[str, str]]:
    if baseline_dir is None:
        return {}
    baseline_path = baseline_dir / "results.csv"
    if not baseline_path.exists():
        return {}
    with baseline_path.open(newline="", encoding="utf-8") as handle:
        return {row["kernel"]: row for row in csv.DictReader(handle)}


def _with_regression(row: EvalRow, baseline_row: dict[str, str] | None) -> EvalRow:
    if baseline_row is None:
        return row

    regression = ""
    if baseline_row.get("passed") == "true" and not row.passed:
        regression = "pass_to_fail"
    else:
        previous_speedup = _parse_float(baseline_row.get("speedup_vs_torch_compile"))
        current_speedup = row.speedup_vs_torch_compile
        if (
            previous_speedup is not None
            and current_speedup is not None
            and current_speedup < previous_speedup
        ):
            regression = f"speedup_drop:{previous_speedup:.2f}->{current_speedup:.2f}"

    if not regression:
        return row
    return EvalRow(**{**asdict(row), "regression": regression})


def _write_csv(path: Path, rows: list[EvalRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_to_csv(row))


def _write_markdown(path: Path, rows: list[EvalRow]) -> None:
    passed = sum(1 for row in rows if row.passed)
    lines = [
        "# CUDA Engine Eval Summary",
        "",
        f"Pass rate: {passed}/{len(rows)}",
        "",
        "| Kernel | Status | Speedup vs torch.compile | Regression |",
        "|---|---|---:|---|",
    ]
    for row in rows:
        status = "PASS" if row.passed else "FAIL"
        lines.append(
            f"| {row.kernel} | {status} | {_format_float(row.speedup_vs_torch_compile)} | "
            f"{row.regression} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row_to_csv(row: EvalRow) -> dict[str, str]:
    return {
        "kernel": row.kernel,
        "passed": str(row.passed).lower(),
        "run_id": row.run_id,
        "failed_stage": "" if row.failed_stage is None else str(row.failed_stage),
        "failure_reason": row.failure_reason,
        "speedup_vs_torch_compile": _format_float(row.speedup_vs_torch_compile),
        "speedup_vs_reference": _format_float(row.speedup_vs_reference),
        "below_target": "" if row.below_target is None else str(row.below_target).lower(),
        "artifacts_dir": row.artifacts_dir,
        "regression": row.regression,
    }


def _row_to_json(row: EvalRow) -> dict[str, object]:
    return asdict(row)


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"
