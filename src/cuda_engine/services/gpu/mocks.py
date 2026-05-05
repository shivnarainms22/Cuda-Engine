from pathlib import Path
from typing import Any

from cuda_engine.services.gpu.base import (
    BenchmarkResult,
    CompileResult,
    GPURunner,
    NsightMetrics,
    RunResult,
)


class MockGPURunner(GPURunner):
    def __init__(
        self,
        compile_results: list[CompileResult] | None = None,
        run_results: list[RunResult] | None = None,
        benchmark_results: list[BenchmarkResult] | None = None,
        profile_results: list[NsightMetrics] | None = None,
    ) -> None:
        self._compile = list(compile_results or [])
        self._run = list(run_results or [])
        self._benchmark = list(benchmark_results or [])
        self._profile = list(profile_results or [])

    def compile(
        self,
        src: str,
        *,
        target_arch: str,
        extra_flags: tuple[str, ...] = (),
    ) -> CompileResult:
        if not self._compile:
            return CompileResult(ok=True, so_path=Path("/tmp/mock.so"), log="ok")
        return self._compile.pop(0)

    def run_kernel(
        self,
        so_path: Path,
        inputs: list[Any],
        timeout_seconds: int = 30,
    ) -> RunResult:
        if not self._run:
            return RunResult(ok=True, wall_seconds=0.0)
        return self._run.pop(0)

    def benchmark_kernel(
        self,
        so_path: Path,
        inputs: list[Any],
        *,
        warmup_iterations: int = 10,
        timed_iterations: int = 50,
        timeout_seconds: int = 60,
    ) -> BenchmarkResult:
        if not self._benchmark:
            return BenchmarkResult(
                ok=True,
                custom_ms=1.0,
                baseline_ms=1.0,
                warmup_iterations=warmup_iterations,
                timed_iterations=timed_iterations,
            )
        return self._benchmark.pop(0)

    def profile(self, so_path: Path, inputs: list[Any]) -> NsightMetrics:
        if not self._profile:
            return NsightMetrics(occupancy=0.5, regs_per_thread=64)
        return self._profile.pop(0)
