from importlib import import_module
from typing import Any


def load_target_caps(target_arch: str) -> dict[str, Any]:
    module = import_module(f"cuda_engine.targets.{target_arch}")
    return dict(module.CAPS)
