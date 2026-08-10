"""Drive the full export -> install -> use loop for one run, on real hardware.

    python tools/export_validation/validate_export.py --run-id <id> [--runs-root DIR]

Steps, in order:
  1. ``cuda-engine export`` the run into a temp directory
  2. ``pip install`` that package
  3. run ``standalone_check.py`` in a FRESH subprocess that has never imported
     ``cuda_engine``

Step 3 is the point. Everything before it is already covered by the unit suite;
only a fresh process on a GPU can show that the package actually builds, loads,
computes correctly, and traces under ``torch.compile``.

Costs nothing beyond the GPU: it consumes an existing run and makes no LLM calls.
Exit code 0 means the loop works end to end.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).parent


def _perf_args(run_dir: Path) -> list[str]:
    """Build the perf-comparison arguments from what the run itself recorded.

    The benchmark shape comes from the engine's own ``_benchmark_shape`` rather than
    being re-derived here, and the timings come from the run's ``benchmark.json``, so
    the exported kernel is compared against the original measurement on its own terms.
    Unlike the standalone check, this driver may import cuda_engine.
    """
    bench_path = run_dir / "stage4_performance" / "benchmark.json"
    checkpoint_path = run_dir / "checkpoint.json"
    if not bench_path.is_file() or not checkpoint_path.is_file():
        print("  (no stage4_performance/benchmark.json -- skipping perf comparison)")
        return []

    try:
        from cuda_engine.models import KernelSpec
        from cuda_engine.stages.performance import _benchmark_shape
    except ImportError:
        print("  (cuda_engine unavailable -- skipping perf comparison)")
        return []

    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    spec_raw = json.loads(checkpoint_path.read_text(encoding="utf-8")).get("objects", {}).get("spec")
    if spec_raw is None:
        return []

    settings = bench.get("settings") or {}
    total = settings.get("performance_shape_n")
    if not isinstance(total, int):
        print("  (run did not record performance_shape_n -- skipping perf comparison)")
        return []

    shape = _benchmark_shape(KernelSpec.model_validate(spec_raw), total_elements=total)
    out = ["--bench-size", str(shape[0])]
    for flag, key in (("--warmup", "benchmark_warmup_iterations"), ("--iters", "benchmark_timed_iterations")):
        if isinstance(settings.get(key), int):
            out += [flag, str(settings[key])]
    if isinstance(bench.get("custom_ms"), (int, float)):
        out += ["--expect-ms", str(bench["custom_ms"])]
    return out


def _cli() -> list[str]:
    """The cuda-engine CLI, preferring the console script a user would actually run."""
    script = shutil.which("cuda-engine")
    return [script] if script else [sys.executable, "-m", "cuda_engine.cli"]


def _run(cmd: list[str], *, label: str) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {result.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Artifact root holding the run. Defaults to ~/.cache/cuda_engine/runs/.",
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--keep", action="store_true", help="Keep the exported package directory.")
    parser.add_argument(
        "--no-perf",
        action="store_true",
        help="Skip re-measuring performance against the original run.",
    )
    args = parser.parse_args()

    runs_root = args.runs_root or (Path.home() / ".cache" / "cuda_engine" / "runs")
    run_dir = runs_root / args.run_id
    if not run_dir.is_dir():
        raise SystemExit(f"run directory not found: {run_dir}")

    reference = run_dir / "inputs" / "reference.py"
    if not reference.is_file():
        raise SystemExit(f"run has no reference.py at {reference}; cannot verify correctness")

    workdir = Path(tempfile.mkdtemp(prefix="ce-export-"))
    pkg_dir = workdir / "pkg"

    print("=" * 72)
    print(f"export validation -- run {args.run_id}")
    print("=" * 72)

    _run(
        [*_cli(), "export", args.run_id, "--runs-root", str(runs_root), "--out", str(pkg_dir)],
        label="export",
    )

    # Do not trust the exit code alone: a CLI invoked the wrong way can exit 0 and do
    # nothing. Confirm the artifact exists before continuing.
    manifests = list(pkg_dir.glob("ce_*/manifest.json")) if pkg_dir.is_dir() else []
    if not manifests:
        raise SystemExit(
            f"export reported success but produced no package under {pkg_dir}; "
            f"nothing to validate"
        )
    manifest_path = manifests[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    package = manifest["package"]
    print(f"\nexported package: {package}")
    print(f"  shipped kernel : {manifest['source_kernel']}")
    print(f"  nvcc flags     : {manifest['nvcc_flags']}")
    print(f"  verified       : {manifest['verified']}")

    _run([sys.executable, "-m", "pip", "install", "--no-build-isolation", str(pkg_dir)], label="pip install")

    perf_args = _perf_args(run_dir) if not args.no_perf else []
    if perf_args:
        print(f"  perf compare   : {' '.join(perf_args)}")

    print("\n--- standalone check (fresh process, no cuda_engine) ---")
    check = subprocess.run(
        [sys.executable, str(_HERE / "standalone_check.py"),
         "--package", package, "--reference", str(reference), "--size", str(args.size),
         *perf_args],
        cwd=str(workdir),  # not the repo root, so `import cuda_engine` cannot resolve locally
        check=False,
    )
    if check.returncode != 0:
        print("\nEXPORT VALIDATION FAILED", flush=True)
        return check.returncode

    if not args.keep:
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        print(f"\nkept: {pkg_dir}")

    print("\nEXPORT VALIDATION PASSED -- the export loop works end to end.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
