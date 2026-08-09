"""Export a verified run as a self-contained, importable kernel package.

A successful run leaves a ``.cu`` inside the run directory. That is a result, not
a dependency: using it means hand-writing a ``cpp_extension.load`` call,
hand-registering a fake impl, and re-deriving from JSON what was actually
verified. This module turns a run into a package a consumer can install, import,
call, and ``torch.compile`` -- carrying its own evidence.

The core (:func:`build_export`) is pure: it maps relative path -> file content and
touches no filesystem, so the whole layout is testable without a GPU.

See ``docs/superpowers/specs/2026-08-01-kernel-export-design.md``.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cuda_engine.models import CorrectnessReport, KernelSpec, PerformanceReport
from cuda_engine.services.store.base import ArtifactStore
from cuda_engine.torch_compat import render_fake_module

__all__ = ["ExportError", "build_export", "package_name", "write_export"]

#: Candidate kernel sources, most-preferred first. The polished kernel only exists
#: when the annotated source recompiled *and* re-passed correctness.
_KERNEL_CANDIDATES = (
    "stage5_polish/final/kernel.cu",
    "stage2_codegen/kernel.cu",
)

_PREBUILT_SO = "stage5_polish/final/kernel.so"

#: Mirrors ``SynthesisConfig.nvcc_flags``. Used when a run predates config capture;
#: a package built with different flags than were measured is not the same kernel.
_DEFAULT_NVCC_FLAGS = ["-O3", "--use_fast_math"]


class ExportError(RuntimeError):
    """A run cannot be exported as a kernel package."""


def package_name(spec_name: str) -> str:
    """Sanitise a spec name into an importable module name.

    Prefixed ``ce_`` so an exported package cannot shadow a module the consumer
    already has (a kernel called ``math`` would otherwise be a nasty surprise).
    """
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", spec_name).strip("_").lower()
    if not cleaned:
        cleaned = "kernel"
    if cleaned[0].isdigit():
        cleaned = f"k{cleaned}"
    return f"ce_{cleaned}"


def _read_json(store: ArtifactStore, run_id: str, rel_path: str) -> Any:
    try:
        return store.read_json(run_id, rel_path)
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"cannot read {rel_path} for run {run_id!r}: {exc}") from exc


def _load_spec(store: ArtifactStore, run_id: str) -> KernelSpec:
    checkpoint = _read_json(store, run_id, "checkpoint.json")
    raw = (checkpoint or {}).get("objects", {}).get("spec")
    if raw is None:
        raise ExportError(f"run {run_id!r} has no frozen KernelSpec; cannot export")
    try:
        return KernelSpec.model_validate(raw)
    except ValueError as exc:
        raise ExportError(f"run {run_id!r} has an unreadable KernelSpec: {exc}") from exc


def _select_kernel(store: ArtifactStore, run_id: str) -> tuple[str, str]:
    for rel_path in _KERNEL_CANDIDATES:
        if store.exists(run_id, rel_path):
            return rel_path, store.read_text(run_id, rel_path)
    raise ExportError(
        f"run {run_id!r} has no kernel source at any of {list(_KERNEL_CANDIDATES)}"
    )


def build_export(
    store: ArtifactStore,
    run_id: str,
    *,
    force: bool = False,
) -> dict[str, str]:
    """Build the text files of a kernel package for ``run_id``.

    Returns a mapping of package-relative path -> file content. Binary artifacts
    (a prebuilt ``.so``) are handled by :func:`write_export`, keeping this pure.

    Raises :class:`ExportError` unless the run's correctness gate passed, or
    ``force`` is set -- in which case the evidence file carries an UNVERIFIED
    banner. There is no path to an unmarked unverified package.
    """
    report = _read_json(store, run_id, "report.json")
    correctness = _coerce(CorrectnessReport, report.get("correctness"))
    performance = _coerce(PerformanceReport, report.get("performance"))

    verified = bool(correctness and correctness.passed)
    if not verified and not force:
        raise ExportError(
            f"run {run_id!r} did not pass its correctness gate; refusing to export an "
            f"unverified kernel (use force=True to override, which stamps the package "
            f"UNVERIFIED)"
        )

    spec = _load_spec(store, run_id)
    source_kernel, kernel_src = _select_kernel(store, run_id)
    prebuilt_so = store.exists(run_id, _PREBUILT_SO)
    nvcc_flags = _nvcc_flags(store, run_id)
    pkg = package_name(spec.name)

    manifest = {
        "kernel_name": spec.name,
        "package": pkg,
        "run_id": run_id,
        "target_arch": spec.target_arch,
        "source_kernel": source_kernel,
        "prebuilt_so": prebuilt_so,
        "nvcc_flags": nvcc_flags,
        "verified": verified,
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "generator": "cuda-engine",
    }

    return {
        "pyproject.toml": _render_pyproject(pkg, spec),
        "README.md": _render_readme(pkg, spec, performance, verified=verified),
        "VERIFICATION.md": _render_verification(
            spec, correctness, performance, manifest, verified=verified
        ),
        f"{pkg}/__init__.py": _render_init(pkg, spec, nvcc_flags),
        f"{pkg}/fake_impl.py": render_fake_module(spec),
        f"{pkg}/kernel.cu": kernel_src,
        f"{pkg}/spec.json": spec.model_dump_json(indent=2),
        f"{pkg}/manifest.json": json.dumps(manifest, indent=2),
    }


def write_export(
    store: ArtifactStore,
    run_id: str,
    out_dir: Path,
    *,
    force: bool = False,
) -> Path:
    """Write a kernel package for ``run_id`` into ``out_dir``.

    Refuses a non-empty destination: an export is a coherent package, and merging
    it into unrelated files silently produces something that is neither.
    """
    files = build_export(store, run_id, force=force)
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ExportError(f"destination {out_dir} is not empty")

    for rel_path, content in files.items():
        target = out_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    if store.exists(run_id, _PREBUILT_SO):
        spec = _load_spec(store, run_id)
        so_target = out_dir / package_name(spec.name) / "kernel.so"
        so_target.write_bytes(store.read_bytes(run_id, _PREBUILT_SO))
    return out_dir


def _nvcc_flags(store: ArtifactStore, run_id: str) -> list[str]:
    """The nvcc flags the kernel was actually compiled and measured with.

    Falls back to the engine defaults for runs that predate config capture --
    silently building the export with torch's defaults would mean shipping a
    differently-compiled kernel than the one VERIFICATION.md describes.
    """
    if not store.exists(run_id, "inputs/config.json"):
        return list(_DEFAULT_NVCC_FLAGS)
    try:
        config = store.read_json(run_id, "inputs/config.json")
    except (OSError, json.JSONDecodeError):
        return list(_DEFAULT_NVCC_FLAGS)
    flags = (config or {}).get("nvcc_flags")
    if not isinstance(flags, list) or not all(isinstance(f, str) for f in flags):
        return list(_DEFAULT_NVCC_FLAGS)
    return list(flags)


def _coerce(model: type[Any], raw: Any) -> Any:
    if raw is None:
        return None
    try:
        return model.model_validate(raw)
    except ValueError:
        return None


def _render_pyproject(pkg: str, spec: KernelSpec) -> str:
    dist = pkg.replace("_", "-")
    return f'''\
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "{dist}"
version = "0.1.0"
description = "CUDA kernel {spec.name!r} ({spec.target_arch}), generated and verified by cuda-engine"
requires-python = ">=3.11"
dependencies = ["torch>=2.4"]

[tool.setuptools]
packages = ["{pkg}"]

[tool.setuptools.package-data]
"{pkg}" = ["kernel.cu", "kernel.so", "spec.json", "manifest.json"]
'''


def _render_init(pkg: str, spec: KernelSpec, nvcc_flags: list[str]) -> str:
    return f'''\
"""CUDA kernel {spec.name!r} for {spec.target_arch}, generated by cuda-engine.

Importing this package loads the kernel and registers its fake (meta)
implementation, so ``forward`` composes inside ``torch.compile`` without a graph
break. See VERIFICATION.md for what was verified and what was not.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import torch

from . import fake_impl

_HERE = Path(__file__).parent
_OP_NAME = "cuda_engine::forward"
#: The flags this kernel was compiled and measured with. Building with anything
#: else means the installed kernel is not the one VERIFICATION.md describes.
_NVCC_FLAGS = {nvcc_flags!r}
_lock = threading.Lock()
_loaded = False


def _load() -> None:
    """Load the kernel, preferring a prebuilt .so and falling back to a source build.

    The prebuilt .so is tied to the exact torch/CUDA build it came from; when it
    does not load, building from source is the honest fallback rather than a
    confusing symbol error.
    """
    global _loaded
    with _lock:
        if _loaded:
            return
        so_path = _HERE / "kernel.so"
        if so_path.is_file():
            try:
                torch.ops.load_library(str(so_path))
                _loaded = True
            except OSError:
                pass
        if not _loaded:
            from torch.utils.cpp_extension import load

            load(
                name="{pkg}_ext",
                sources=[str(_HERE / "kernel.cu")],
                extra_cuda_cflags=list(_NVCC_FLAGS),
                is_python_module=False,
                verbose=False,
            )
            _loaded = True
        try:
            fake_impl.register(_OP_NAME)
        except RuntimeError:
            # Already registered in this process (re-import, or another copy).
            pass


def forward(*args: Any) -> Any:
    """Call the kernel. Loads and registers on first use."""
    if not _loaded:
        _load()
    return torch.ops.cuda_engine.forward(*args)


__all__ = ["forward"]
'''


def _render_readme(
    pkg: str,
    spec: KernelSpec,
    performance: PerformanceReport | None,
    *,
    verified: bool,
) -> str:
    speedup = _fmt(performance.speedup_vs_torch_compile if performance else None)
    banner = "" if verified else "\n> **UNVERIFIED** — exported with `--force`. See VERIFICATION.md.\n"
    return f'''# {pkg}

CUDA kernel `{spec.name}` for `{spec.target_arch}`, generated and verified by
[cuda-engine](https://github.com/shivnarainms22/Cuda-Engine).
{banner}
Measured **{speedup}×** vs the fastest `torch.compile` mode. Full evidence, including
what was *not* verified, is in [VERIFICATION.md](VERIFICATION.md).

## Install

```bash
pip install .
```

Requires a CUDA toolchain (`nvcc`) and PyTorch 2.4+. The kernel builds from source
on first use if the bundled `.so` does not match your environment.

## Use

```python
import torch
from {pkg} import forward

out = forward(*inputs)
```

It registers a fake (meta) implementation, so it composes inside `torch.compile`
without a graph break:

```python
compiled = torch.compile(lambda *xs: forward(*xs), fullgraph=True)
```

## Contract

| | name | dtype | shape |
|---|---|---|---|
{_arg_rows(spec)}

Tolerance: `rtol={spec.precision_tolerance.rtol}`, `atol={spec.precision_tolerance.atol}`.
'''


def _arg_rows(spec: KernelSpec) -> str:
    rows = [f"| input | `{a.name}` | {a.dtype} | `{list(a.shape)}` |" for a in spec.inputs]
    rows += [f"| output | `{a.name}` | {a.dtype} | `{list(a.shape)}` |" for a in spec.outputs]
    return "\n".join(rows)


def _render_verification(
    spec: KernelSpec,
    correctness: CorrectnessReport | None,
    performance: PerformanceReport | None,
    manifest: dict[str, Any],
    *,
    verified: bool,
) -> str:
    banner = (
        ""
        if verified
        else "> ## ⚠️ UNVERIFIED\n>\n> This package was exported with `force` despite its\n"
        "> correctness gate not passing. Do not rely on it.\n\n"
    )
    shapes = ", ".join(str(tuple(s)) for s in correctness.shapes_tested) if correctness else "none"
    tc = performance.speedup_vs_torch_compile if performance else None
    eager = performance.speedup_vs_reference if performance else None
    below = bool(performance and performance.below_target)
    perf_note = (
        f"\n**Below the performance target.** Measured {_fmt(tc)}× vs `torch.compile`; "
        f"the kernel is correct but not faster than the baseline on the measured shape.\n"
        if below
        else ""
    )
    notes = "\n".join(f"- {n}" for n in (performance.notes if performance else [])) or "- none"

    return f'''# Verification — `{spec.name}`

{banner}Generated by cuda-engine from run `{manifest["run_id"]}` on
{manifest["exported_at"]}. Shipped kernel: `{manifest["source_kernel"]}`.

## Correctness

| | |
|---|---|
| Gate passed | **{_yesno(correctness.passed if correctness else False)}** |
| Shapes tested | {shapes} |
| Max absolute error | {_fmt(correctness.max_abs_err if correctness else None, 3)} |
| Max relative error | {_fmt(correctness.max_rel_err if correctness else None, 3)} |
| Tolerance | rtol={spec.precision_tolerance.rtol}, atol={spec.precision_tolerance.atol} |

Outputs were compared elementwise against the PyTorch reference at every shape
listed above, on real hardware.

## Performance
{perf_note}
| | |
|---|---|
| vs fastest `torch.compile` | {_fmt(tc)}× |
| vs eager reference | {_fmt(eager)}× |
| Achieved bandwidth | {_fmt(performance.achieved_gbps if performance else None, 1)} GB/s |

The baseline is the *fastest* of `torch.compile`'s modes (`default` /
`max-autotune-no-cudagraphs` / `reduce-overhead`), not the first one tried, so a
win here means beating torch.compile at its best.

Run notes:
{notes}

## Not verified

Stating the negative space is the point of this document.

- **Architecture.** Runtime verification was performed on `{manifest["target_arch"]}`
  only. Other architectures are untested, including any listed as codegen targets.
- **Shapes.** Only the shapes listed above were checked. Behaviour at other shapes
  — in particular much larger ones, where 32-bit index overflow and tiling-edge
  bugs appear — is not covered unless that shape is in the list.
- **Dtypes and layouts.** Only the contract in `spec.json`. Non-contiguous or
  transposed inputs were not tested.
- **Numerics beyond tolerance.** Agreement is within the tolerance above, not
  bitwise. Accumulation order differs from the reference.
- **Concurrency and streams.** Not tested under multiple streams or graph capture.
- **Backward pass.** Forward only. No autograd formula is registered.
'''


def _yesno(value: bool) -> str:
    return "yes" if value else "NO"


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}g}" if digits == 3 else f"{value:.{digits}f}"
