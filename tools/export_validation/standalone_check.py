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


#: How much slower than the original run the exported kernel may be before this is
#: called a regression. Generous enough to absorb driver/torch differences between
#: the run and now; a genuine build problem (e.g. losing -O3) is far worse than 25%.
_PERF_REGRESSION_FACTOR = 1.25


class CheckFailed(Exception):
    """A validation step failed."""


def _time_ms(torch: Any, call: Any, *, warmup: int, iters: int) -> float:
    """Median wall time of `call` in milliseconds, with the GPU synchronised."""
    import time

    for _ in range(max(warmup, 0)):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(max(iters, 1)):
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


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


def _err_stats(torch: Any, actual: Any, expected: Any) -> tuple[float, float]:
    """Max absolute and relative error, so the report carries magnitudes not a boolean."""
    max_abs, max_rel = 0.0, 0.0
    for x, y in zip(_as_list(actual), _as_list(expected), strict=False):
        if not x.is_floating_point():
            continue
        diff = (x.float() - y.float().to(x.device)).abs()
        if diff.numel() == 0:
            continue
        max_abs = max(max_abs, float(diff.max().item()))
        denom = y.float().to(x.device).abs().clamp_min(1e-12)
        max_rel = max(max_rel, float((diff / denom).max().item()))
    return max_abs, max_rel


def _perturb(torch: Any, expected: Any, *, rtol: float, atol: float) -> list[Any]:
    """A wrong answer that MUST fall outside the tolerance band.

    A fixed additive nudge does not work: `allclose` compares against
    `atol + rtol * |expected|`, so on a GEMM whose outputs are ~1e12 with
    rtol=1e-3 the band is ~1e9 and adding 1.0 is invisible. The perturbation has
    to scale with the value, plus an absolute floor for outputs that are all zero.
    """
    out = []
    for t in _as_list(expected):
        if t.is_floating_point():
            out.append(t * (1.0 + 1000.0 * max(rtol, 1e-9)) + 1000.0 * max(atol, 1e-9))
        else:
            out.append(t + 1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, help="Installed package name, e.g. ce_rms_norm_fp16")
    parser.add_argument("--reference", required=True, type=Path, help="Path to the run's reference.py")
    parser.add_argument("--size", type=int, default=256, help="Size bound to every symbolic dim")
    parser.add_argument(
        "--bench-size",
        type=int,
        default=None,
        help="Size bound to every symbolic dim for the perf re-measurement. Omit to skip.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--expect-ms",
        type=float,
        default=None,
        help="custom_ms the original run recorded, for regression comparison.",
    )
    parser.add_argument("--expect-eager-ms", type=float, default=None)
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
    max_abs, max_rel = _err_stats(torch, out, expected)
    if not _matches(torch, out, expected, rtol=rtol, atol=atol):
        raise CheckFailed(
            f"output does not match the reference within rtol={rtol} atol={atol} "
            f"(max_abs_err={max_abs:.3g}, max_rel_err={max_rel:.3g})"
        )
    # Report magnitudes, not a boolean. On a GEMM with large outputs the absolute
    # error is huge and meaningless; the relative error is the number that matters.
    record(
        "matches reference",
        f"max_rel_err={max_rel:.3g} (rtol={rtol}), max_abs_err={max_abs:.3g}, size={args.size}",
    )

    # 3b. control: the comparison must be able to fail, or step 3 proves nothing
    perturbed = _perturb(torch, expected, rtol=rtol, atol=atol)
    if _matches(torch, out, perturbed, rtol=rtol, atol=atol):
        raise CheckFailed(
            "control failed: the comparison accepts a deliberately wrong answer, so the "
            "correctness pass above is vacuous at this magnitude"
        )
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

    # 5. performance: is the kernel you installed still the kernel that was measured?
    #    VERIFICATION.md inherits its speedup from the original run. Without this,
    #    an export that builds differently and runs slower passes silently.
    if args.bench_size is not None:
        bench_inputs = _make_inputs(torch, spec, args.bench_size, "cuda")
        bench_expected = reference(*bench_inputs)
        bench_out = pkg.forward(*bench_inputs)
        _, b_rel = _err_stats(torch, bench_out, bench_expected)
        if not _matches(torch, bench_out, bench_expected, rtol=rtol, atol=atol):
            raise CheckFailed(
                f"kernel is wrong at the benchmark shape {args.bench_size} "
                f"(max_rel_err={b_rel:.3g}) -- a speedup here would be meaningless"
            )
        record("correct at the benchmark shape", f"size={args.bench_size} max_rel_err={b_rel:.3g}")

        kernel_ms = _time_ms(
            torch, lambda: pkg.forward(*bench_inputs), warmup=args.warmup, iters=args.iters
        )
        eager_ms = _time_ms(
            torch, lambda: reference(*bench_inputs), warmup=args.warmup, iters=args.iters
        )
        detail = f"kernel={kernel_ms:.3f}ms eager={eager_ms:.3f}ms ({eager_ms / kernel_ms:.2f}x)"

        if args.expect_ms is None:
            record("performance re-measured", detail + " -- no recorded custom_ms to compare")
        else:
            ratio = kernel_ms / args.expect_ms
            detail += f" | run recorded {args.expect_ms:.3f}ms -> {ratio:.2f}x of it"
            if ratio > _PERF_REGRESSION_FACTOR:
                raise CheckFailed(
                    f"exported kernel is {ratio:.2f}x slower than the original run "
                    f"({kernel_ms:.3f}ms vs {args.expect_ms:.3f}ms); the installed package "
                    f"is not performing like the one VERIFICATION.md describes. {detail}"
                )
            record("performance holds vs the original run", detail)

    print(f"\nALL {len(results)} CHECKS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckFailed as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
