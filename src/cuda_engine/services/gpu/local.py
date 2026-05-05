from pathlib import Path
from typing import Any

from cuda_engine.config import SynthesisConfig
from cuda_engine.services.gpu.base import CompileResult, GPURunner, NsightMetrics, RunResult


class LocalGPURunner(GPURunner):
    def __init__(self, cfg: SynthesisConfig | None = None) -> None:
        self.cfg = cfg

    def compile(
        self,
        src: str,
        *,
        target_arch: str,
        extra_flags: tuple[str, ...] = (),
    ) -> CompileResult:
        raise NotImplementedError("LocalGPURunner lands in M1")

    def run_kernel(
        self,
        so_path: Path,
        inputs: list[Any],
        timeout_seconds: int = 30,
    ) -> RunResult:
        raise NotImplementedError("LocalGPURunner lands in M1")

    def profile(self, so_path: Path, inputs: list[Any]) -> NsightMetrics:
        raise NotImplementedError("LocalGPURunner lands in M1")
