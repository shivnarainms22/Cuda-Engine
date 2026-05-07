from collections.abc import Callable
from typing import Any

from cuda_engine.models import CorrectnessReport, KernelArtifact, KernelSpec
from cuda_engine.stages.base import Stage

CORRECTNESS_SHAPES: tuple[tuple[int, ...], ...] = ((0,), (1,), (127,), (128,), (1024,), (4097,))


class Stage3Correctness(Stage):
    name = "correctness"

    def run(
        self,
        *,
        spec: KernelSpec,
        artifact: KernelArtifact,
        reference: Callable[..., Any],
        run_id: str,
        retry_budget: int = 3,
        artifact_prefix: str = "stage3_correctness",
        correctness_shapes: tuple[tuple[int, ...], ...] = CORRECTNESS_SHAPES,
    ) -> CorrectnessReport:
        if self.gpu is None or self.store is None:
            raise RuntimeError("Stage3Correctness requires gpu and store services")
        if artifact.kernel_so_path is None:
            return CorrectnessReport(
                passed=False,
                max_abs_err=float("inf"),
                max_rel_err=float("inf"),
                shapes_tested=[],
                shape_results=[],
                failing_inputs=[{"error": "kernel_so_path is required for correctness"}],
            )

        max_abs_err = 0.0
        max_rel_err = 0.0
        shape_results: list[dict[str, Any]] = []
        failing_inputs: list[dict[str, Any]] = []
        for shape in correctness_shapes:
            inputs = _make_inputs(spec, shape=shape)
            expected = _as_output_list(reference(*inputs))
            run_result = self.gpu.run_kernel(artifact.kernel_so_path, inputs)
            shape_result, failures = _evaluate_shape(
                shape=shape,
                expected=expected,
                run_result=run_result,
                rtol=spec.precision_tolerance.rtol,
                atol=spec.precision_tolerance.atol,
            )
            shape_results.append(shape_result)
            failing_inputs.extend(failures)
            max_abs_err = max(max_abs_err, float(shape_result["max_abs_err"]))
            max_rel_err = max(max_rel_err, float(shape_result["max_rel_err"]))

        report = CorrectnessReport(
            passed=not failing_inputs,
            max_abs_err=max_abs_err,
            max_rel_err=max_rel_err,
            shapes_tested=list(correctness_shapes),
            shape_results=shape_results,
            failing_inputs=failing_inputs,
        )
        self.store.write_json(run_id, f"{artifact_prefix}/report.json", report.model_dump(mode="json"))
        return report


def _make_inputs(spec: KernelSpec, *, shape: tuple[int, ...]) -> list[Any]:
    torch = _torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs: list[Any] = []
    for index, arg in enumerate(spec.inputs):
        tensor_shape = _concrete_shape(arg.shape, fallback=shape)
        if arg.dtype in {"fp32", "fp16", "bf16", "fp64"}:
            dtype = getattr(torch, _torch_dtype_name(arg.dtype))
            value = _floating_input_values(torch, tensor_shape=tensor_shape, dtype=arg.dtype)
            inputs.append(value.to(dtype=dtype, device=device) + index)
        elif arg.dtype in {"int32", "int64", "uint8", "int8"}:
            dtype = getattr(torch, _torch_dtype_name(arg.dtype))
            inputs.append(
                torch.arange(_numel(tensor_shape), dtype=dtype).reshape(tensor_shape).to(device=device)
            )
        else:
            raise ValueError(f"Unsupported dtype for correctness input generation: {arg.dtype}")
    return inputs


def _floating_input_values(torch: Any, *, tensor_shape: tuple[int, ...], dtype: str) -> Any:
    values = torch.arange(_numel(tensor_shape), dtype=torch.float32).reshape(tensor_shape)
    if dtype in {"fp16", "bf16"}:
        return (values.remainder(17) - 8) / 8
    return values


def _concrete_shape(symbolic_shape: tuple[str, ...], *, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if not symbolic_shape:
        return ()
    values: list[int] = []
    symbol_values: dict[str, int] = {}
    next_fallback_index = 0
    for dim in symbolic_shape:
        try:
            values.append(int(dim))
        except ValueError:
            if dim not in symbol_values:
                fallback_index = min(next_fallback_index, len(fallback) - 1)
                symbol_values[dim] = fallback[fallback_index]
                next_fallback_index += 1
            values.append(symbol_values[dim])
    return tuple(values)


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def _torch_dtype_name(dtype: str) -> str:
    return {
        "fp32": "float32",
        "fp16": "float16",
        "bf16": "bfloat16",
        "fp64": "float64",
        "int32": "int32",
        "int64": "int64",
        "uint8": "uint8",
        "int8": "int8",
    }[dtype]


def _as_output_list(value: Any) -> list[Any]:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def _evaluate_shape(
    *,
    shape: tuple[int, ...],
    expected: list[Any],
    run_result: Any,
    rtol: float,
    atol: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not run_result.ok or run_result.output_tensors is None:
        failure = {"shape": shape, "error": run_result.stderr or "kernel run failed"}
        return (
            {
                "shape": shape,
                "passed": False,
                "max_abs_err": float("inf"),
                "max_rel_err": float("inf"),
                "error": failure["error"],
            },
            [failure],
        )

    max_abs_err = 0.0
    max_rel_err = 0.0
    failures: list[dict[str, Any]] = []
    for output_index, (actual, exp) in enumerate(zip(run_result.output_tensors, expected, strict=False)):
        actual_shape = tuple(getattr(actual, "shape", ()))
        expected_shape = tuple(getattr(exp, "shape", ()))
        if actual_shape != expected_shape:
            failures.append(
                {
                    "shape": shape,
                    "output_index": output_index,
                    "error": f"shape mismatch: expected {expected_shape}, got {actual_shape}",
                }
            )
            max_abs_err = float("inf")
            max_rel_err = float("inf")
            continue
        abs_err, rel_err = _error_stats(actual, exp)
        max_abs_err = max(max_abs_err, abs_err)
        max_rel_err = max(max_rel_err, rel_err)
        if not _within_tolerance(actual, exp, rtol=rtol, atol=atol):
            failures.append(
                {
                    "shape": shape,
                    "output_index": output_index,
                    "max_abs_err": abs_err,
                    "max_rel_err": rel_err,
                }
            )

    if len(run_result.output_tensors) != len(expected):
        failures.append(
            {
                "shape": shape,
                "error": f"expected {len(expected)} outputs, got {len(run_result.output_tensors)}",
            }
        )

    return (
        {
            "shape": shape,
            "passed": not failures,
            "max_abs_err": max_abs_err,
            "max_rel_err": max_rel_err,
        },
        failures,
    )


def _error_stats(actual: Any, expected: Any) -> tuple[float, float]:
    torch = _torch()
    actual_tensor = _to_tensor(actual)
    expected_tensor = _to_tensor(expected).to(device=actual_tensor.device)
    diff = (actual_tensor - expected_tensor).abs()
    abs_err = float(diff.max().item()) if diff.numel() else 0.0
    denom = expected_tensor.abs().clamp_min(1e-12)
    rel_err = float((diff / denom).max().item()) if diff.numel() else 0.0
    if not torch.isfinite(diff).all():
        return float("inf"), float("inf")
    return abs_err, rel_err


def _within_tolerance(actual: Any, expected: Any, *, rtol: float, atol: float) -> bool:
    torch = _torch()
    actual_tensor = _to_tensor(actual)
    expected_tensor = _to_tensor(expected).to(device=actual_tensor.device)
    return bool(torch.allclose(actual_tensor, expected_tensor, rtol=rtol, atol=atol))


def _to_tensor(value: Any) -> Any:
    torch = _torch()
    if hasattr(value, "detach"):
        return value.detach()
    return torch.as_tensor(value)


def _torch() -> Any:
    import torch

    return torch
