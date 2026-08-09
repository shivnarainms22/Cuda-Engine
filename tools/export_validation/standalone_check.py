"""Validate an installed cuda-engine kernel package on real hardware.

Runs in a FRESH process that has never imported ``cuda_engine``. That is the whole
point: the unit suite proves the package *files* are generated correctly, but only
this proves the package actually builds, loads, computes, and composes inside
``torch.compile`` on a GPU.

It imports only ``torch``, the installed kernel package, and the run's
``reference.py`` (which is plain PyTorch).

    python standalone_check.py --package ce_rms_norm_fp16 --reference /path/reference.py

Exit code 0 means every check passed. Verify by exit code, not by reading output.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

_DTYPES = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
    "fp64": "float64",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "int8": "int8",
}


class CheckFailed(Exception):
    """A validation step failed."""


def _assert_clean_process() -> None:
    """The package must not need the generator that produced it."""
    if "cuda_engine" in sys.modules:
        raise CheckFailed(
            "cuda_engine is already imported in this process; the standalone check "
            "must run somewhere it has never been imported"
        )


def _bind_shapes(decls: list[dict[str, Any]], size: int) -> list[tuple[int, ...]]:
    """Bind every symbolic dim to `size`, honouring literal dims."""
    shapes: list[tuple[int, ...]] = []
    for arg in decls:
        dims: list[int] = []
        for dim in arg["shape"]:
            try:
                dims.append(int(dim))
            except ValueError:
                dims.append(size)
        shapes.append(tuple(dims))
    return shapes


def _make_inputs(torch: Any, spec: dict[str, Any], size: int, device: str) -> list[Any]:
    """Mirror the engine's correctness input generator so the comparison is faithful."""
    inputs = []
    for index, (arg, shape) in enumerate(zip(spec["inputs"], _bind_shapes(spec["inputs"], size), strict=True)):
        dtype = getattr(torch, _DTYPES[arg["dtype"]])
        numel = 1
        for d in shape:
            numel *= d
        if arg["dtype"] in {"fp32", "fp16", "bf16", "fp64"}:
            values = torch.arange(numel, dtype=torch.float32).reshape(shape)
            if arg["dtype"] in {"fp16", "bf16"}:
                values = (values.remainder(17) - 8) / 8
            inputs.append(values.to(dtype=dtype, device=device) + index)
        else:
            inputs.append(torch.arange(numel, dtype=dtype).reshape(shape).to(device=device))
    return inputs


def _load_reference(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("_ce_reference", path)
    if spec is None or spec.loader is None:
        raise CheckFailed(f"cannot import reference from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "REFERENCE", None) or getattr(module, "reference", None)
    if fn is None:
        raise CheckFailed(f"{path} defines neither REFERENCE nor reference()")
    return fn


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (tuple, list)):
        return list(value)
    return [value]


def _matches(torch: Any, actual: Any, expected: Any, *, rtol: float, atol: float) -> bool:
    a, e = _as_list(actual), _as_list(expected)
    if len(a) != len(e):
        return False
    return all(
        x.shape == y.shape and torch.allclose(x, y.to(x.dtype), rtol=rtol, atol=atol)
        for x, y in zip(a, e, strict=True)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, help="Installed package name, e.g. ce_rms_norm_fp16")
    parser.add_argument("--reference", required=True, type=Path, help="Path to the run's reference.py")
    parser.add_argument("--size", type=int, default=256, help="Size bound to every symbolic dim")
    args = parser.parse_args()

    _assert_clean_process()

    import torch

    if not torch.cuda.is_available():
        raise CheckFailed("no CUDA device; this check must run on a GPU")

    results: list[tuple[str, str]] = []

    def record(name: str, detail: str = "") -> None:
        results.append((name, detail))
        print(f"  PASS  {name}" + (f" -- {detail}" if detail else ""), flush=True)

    print(f"device: {torch.cuda.get_device_name(0)}  torch: {torch.__version__}", flush=True)

    # 1. import the installed package (this must not require cuda_engine)
    pkg = importlib.import_module(args.package)
    if "cuda_engine" in sys.modules:
        raise CheckFailed("importing the exported package pulled in cuda_engine; it is not standalone")
    record("package imports without cuda_engine")

    spec = json.loads((Path(pkg.__file__).parent / "spec.json").read_text(encoding="utf-8"))
    manifest = json.loads((Path(pkg.__file__).parent / "manifest.json").read_text(encoding="utf-8"))
    record("manifest read", f"run={manifest['run_id']} flags={manifest['nvcc_flags']}")

    # 2. build/load the kernel and call it -- this is the step that has never run
    inputs = _make_inputs(torch, spec, args.size, "cuda")
    out = pkg.forward(*inputs)
    torch.cuda.synchronize()
    record("kernel builds and runs", f"loaded={'prebuilt .so' if manifest['prebuilt_so'] else 'JIT source build'}")

    # 3. correctness vs the reference
    reference = _load_reference(args.reference)
    expected = reference(*inputs)
    rtol = spec["precision_tolerance"]["rtol"]
    atol = spec["precision_tolerance"]["atol"]
    if not _matches(torch, out, expected, rtol=rtol, atol=atol):
        raise CheckFailed(f"output does not match the reference within rtol={rtol} atol={atol}")
    record("matches reference", f"rtol={rtol} atol={atol} size={args.size}")

    # 3b. control: the comparison must be able to fail, or step 3 proves nothing
    perturbed = [t + 1.0 if t.is_floating_point() else t for t in _as_list(expected)]
    if _matches(torch, out, perturbed, rtol=rtol, atol=atol):
        raise CheckFailed("control failed: comparison accepts a deliberately wrong answer")
    record("control: comparison rejects a wrong answer")

    # 4. the reason the fake impl exists -- no graph break
    def fn(*xs: Any) -> Any:
        result = pkg.forward(*xs)
        first = result[0] if isinstance(result, (tuple, list)) else result
        return first * 2

    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True)
    compiled_out = compiled(*inputs)
    torch.cuda.synchronize()
    record("torch.compile(fullgraph=True) traces it", "no graph break")

    eager_first = _as_list(out)[0] * 2
    if not torch.allclose(compiled_out, eager_first, rtol=rtol, atol=atol):
        raise CheckFailed("compiled result differs from eager")
    record("compiled result matches eager")

    print(f"\nALL {len(results)} CHECKS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckFailed as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
