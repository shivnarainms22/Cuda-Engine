import argparse
import importlib.util
import pickle
import time
import traceback
from pathlib import Path
from typing import Any, cast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--so", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--timed-iterations", type=int, default=50)
    args = parser.parse_args()

    inputs, reference = _load_payload(Path(args.input))

    try:
        if args.benchmark:
            benchmark = _benchmark_forward(
                Path(args.so),
                inputs,
                reference,
                warmup_iterations=args.warmup_iterations,
                timed_iterations=args.timed_iterations,
            )
            payload = {
                "ok": benchmark["ok"],
                "benchmark": benchmark,
                "outputs": None,
                "stdout": "",
                "stderr": benchmark.get("stderr", ""),
            }
        else:
            outputs = _run_forward(Path(args.so), inputs)
            payload = {"ok": True, "outputs": _as_output_list(outputs), "stdout": "", "stderr": ""}
    except Exception:
        payload = {"ok": False, "outputs": None, "stdout": "", "stderr": traceback.format_exc()}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(payload, f)


def _load_payload(input_path: Path) -> tuple[list[Any], Any]:
    """Load pickle payload, supporting both legacy list and new dict formats."""
    with input_path.open("rb") as f:
        raw = pickle.load(f)
    if isinstance(raw, dict):
        reference = raw.get("reference")
        reference_path = raw.get("reference_path")
        if reference is None and reference_path is not None:
            reference = _load_reference_from_path(Path(reference_path))
        return raw.get("inputs", []), reference
    return raw, None


def _load_reference_from_path(path: Path) -> Any:
    """Load the reference callable from a Python source file."""
    spec = importlib.util.spec_from_file_location("_cuda_engine_child_reference", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    return getattr(module, "REFERENCE", None) or getattr(module, "reference", None)


def _run_forward(so_path: Path, inputs: list[Any]) -> Any:
    return _resolve_forward(so_path)(*inputs)


def _resolve_forward(so_path: Path) -> Any:
    python_forward = _try_python_extension_forward(so_path)
    if python_forward is not _NOT_FOUND:
        return python_forward
    return _torch_custom_op_forward(so_path)


def _try_python_extension_forward(so_path: Path, inputs: list[Any] | None = None) -> Any:
    spec = importlib.util.spec_from_file_location("cuda_engine_generated_kernel", so_path)
    loader = getattr(spec, "loader", None)
    if spec is None or loader is None:
        return _NOT_FOUND
    try:
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
    except ImportError:
        return _NOT_FOUND
    forward = getattr(module, "forward", None)
    if forward is None:
        return _NOT_FOUND
    if inputs is not None:
        return forward(*inputs)
    return forward


def _torch_custom_op_forward(so_path: Path) -> Any:
    import torch

    load_library = cast(Any, torch.ops.load_library)
    load_library(str(so_path))
    return cast(Any, torch.ops.cuda_engine.forward)


def _benchmark_forward(
    so_path: Path,
    inputs: list[Any],
    reference: Any,
    *,
    warmup_iterations: int,
    timed_iterations: int,
) -> dict[str, Any]:
    import torch

    use_cuda_events = _has_cuda_inputs(inputs)
    forward = _resolve_forward(so_path)
    for _ in range(warmup_iterations):
        forward(*inputs)
    _synchronize_if_cuda(torch, inputs)
    custom_ms = _time_callable_ms(
        torch,
        lambda: forward(*inputs),
        iterations=timed_iterations,
        use_cuda_events=use_cuda_events,
    )

    baseline_ms: float | None = None
    baseline_mode: str | None = None
    baseline_error: str | None = None
    eager_ms: float | None = None
    if reference is not None:
        eager_ms = _time_eager(
            torch, reference, inputs,
            warmup_iterations=warmup_iterations,
            timed_iterations=timed_iterations,
            use_cuda_events=use_cuda_events,
        )
        baseline_ms, baseline_mode, baseline_error = _measure_torch_compile_baseline(
            torch, reference, inputs,
            warmup_iterations=warmup_iterations,
            timed_iterations=timed_iterations,
        )

    return {
        "ok": True,
        "custom_ms": custom_ms,
        "baseline_ms": baseline_ms,
        "baseline_mode": baseline_mode,
        "baseline_error": baseline_error,
        "eager_ms": eager_ms,
        "achieved_gbps": _achieved_gbps(inputs, custom_ms),
        "warmup_iterations": warmup_iterations,
        "timed_iterations": timed_iterations,
    }


# torch.compile modes tried for the baseline. We benchmark the kernel against the
# FASTEST mode (the honest, hardest bar) rather than the first that compiles.
# reduce-overhead (CUDA graphs) is included but loses to default/max-autotune on
# bandwidth-bound elementwise ops because its static-buffer copies add memory traffic.
_BASELINE_MODES = ("default", "max-autotune-no-cudagraphs", "reduce-overhead")


def _measure_torch_compile_baseline(
    torch: Any,
    reference: Any,
    inputs: list[Any],
    *,
    warmup_iterations: int,
    timed_iterations: int,
) -> tuple[float | None, str | None, str | None]:
    """Time torch.compile(reference) across modes and return the FASTEST.

    Benchmarking against torch.compile's best mode is the honest bar: an
    elementwise kernel that beats only a weak mode (e.g. reduce-overhead, which
    is slow on bandwidth-bound ops) is not really beating torch.compile.

    Returns (baseline_ms, baseline_mode, error_str). On success baseline_ms and
    baseline_mode are set; error_str carries any per-mode failures (non-fatal).
    On total failure baseline_ms and baseline_mode are None and error_str is set.
    """
    use_cuda_events = _has_cuda_inputs(inputs)
    best_ms: float | None = None
    best_mode: str | None = None
    errors: list[str] = []
    for mode in _BASELINE_MODES:
        try:
            _reset_dynamo(torch)
            compiled = torch.compile(reference, mode=mode)
            for _ in range(warmup_iterations):
                compiled(*inputs)
            _synchronize_if_cuda(torch, inputs)
            ms = _time_callable_ms(
                torch,
                lambda _c=compiled: _c(*inputs),
                iterations=timed_iterations,
                use_cuda_events=use_cuda_events,
            )
            if best_ms is None or ms < best_ms:
                best_ms, best_mode = ms, mode
        except Exception as exc:
            errors.append(f"mode={mode!r}: {type(exc).__name__}: {exc}")
    if best_ms is None:
        return None, None, "torch.compile baseline failed: " + "; ".join(errors)
    return best_ms, best_mode, ("; ".join(errors) if errors else None)


def _reset_dynamo(torch: Any) -> None:
    """Clear dynamo's compile cache so each mode actually recompiles."""
    reset = getattr(getattr(torch, "_dynamo", None), "reset", None)
    if callable(reset):
        reset()


def _time_eager(
    torch: Any,
    reference: Any,
    inputs: list[Any],
    *,
    warmup_iterations: int,
    timed_iterations: int,
    use_cuda_events: bool,
) -> float | None:
    """Time the un-compiled eager reference. Returns None if it raises."""
    try:
        for _ in range(warmup_iterations):
            reference(*inputs)
        _synchronize_if_cuda(torch, inputs)
        return _time_callable_ms(
            torch,
            lambda: reference(*inputs),
            iterations=timed_iterations,
            use_cuda_events=use_cuda_events,
        )
    except Exception:
        return None


def _time_callable_ms(torch: Any, action: Any, *, iterations: int, use_cuda_events: bool) -> float:
    if iterations <= 0:
        return 0.0
    if use_cuda_events:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            action()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end)) / iterations

    started_at = time.perf_counter()
    for _ in range(iterations):
        action()
    return (time.perf_counter() - started_at) * 1000.0 / iterations


def _has_cuda_inputs(inputs: list[Any]) -> bool:
    return any(bool(getattr(getattr(tensor, "device", None), "type", None) == "cuda") for tensor in inputs)


def _synchronize_if_cuda(torch: Any, inputs: list[Any]) -> None:
    if _has_cuda_inputs(inputs):
        torch.cuda.synchronize()


def _achieved_gbps(inputs: list[Any], custom_ms: float) -> float | None:
    if custom_ms <= 0 or not inputs or not hasattr(inputs[0], "numel"):
        return None
    input_bytes = sum(_tensor_nbytes(tensor) for tensor in inputs)
    output_bytes = _tensor_nbytes(inputs[0])
    return (input_bytes + output_bytes) / (custom_ms / 1000.0) / 1e9


def _tensor_nbytes(tensor: Any) -> int:
    if not hasattr(tensor, "numel") or not hasattr(tensor, "element_size"):
        return 0
    return int(tensor.numel() * tensor.element_size())


def _as_output_list(outputs: Any) -> list[Any]:
    if isinstance(outputs, tuple):
        return list(outputs)
    if isinstance(outputs, list):
        return outputs
    return [outputs]


class _NotFound:
    pass


_NOT_FOUND = _NotFound()


if __name__ == "__main__":
    main()
