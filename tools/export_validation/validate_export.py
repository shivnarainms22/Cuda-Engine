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

    print("\n--- standalone check (fresh process, no cuda_engine) ---")
    check = subprocess.run(
        [sys.executable, str(_HERE / "standalone_check.py"),
         "--package", package, "--reference", str(reference), "--size", str(args.size)],
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
