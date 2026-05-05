import argparse
import importlib.util
import pickle
import traceback
from pathlib import Path
from typing import Any, cast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--so", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.input).open("rb") as f:
        inputs = pickle.load(f)

    try:
        outputs = _run_forward(Path(args.so), inputs)
        payload = {"ok": True, "outputs": _as_output_list(outputs), "stdout": "", "stderr": ""}
    except Exception:
        payload = {"ok": False, "outputs": None, "stdout": "", "stderr": traceback.format_exc()}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(payload, f)


def _run_forward(so_path: Path, inputs: list[Any]) -> Any:
    module_output = _try_python_extension_forward(so_path, inputs)
    if module_output is not _NOT_FOUND:
        return module_output
    return _run_torch_custom_op(so_path, inputs)


def _try_python_extension_forward(so_path: Path, inputs: list[Any]) -> Any:
    spec = importlib.util.spec_from_file_location("cuda_engine_generated_kernel", so_path)
    if spec is None or spec.loader is None:
        return _NOT_FOUND
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError:
        return _NOT_FOUND
    forward = getattr(module, "forward", None)
    if forward is None:
        return _NOT_FOUND
    return forward(*inputs)


def _run_torch_custom_op(so_path: Path, inputs: list[Any]) -> Any:
    import torch

    load_library = cast(Any, torch.ops.load_library)
    load_library(str(so_path))
    return cast(Any, torch.ops.cuda_engine.forward)(*inputs)


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
