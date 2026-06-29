from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class CompileResult(BaseModel):
    ok: bool
    so_path: Path | None = None
    log: str = ""
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    ptx_size_bytes: int = 0


class RunResult(BaseModel):
    ok: bool
    output_tensors: list[Any] | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    wall_seconds: float = 0.0


class BenchmarkResult(BaseModel):
    ok: bool
    custom_ms: float = 0.0
    baseline_ms: float | None = None
    baseline_mode: str | None = None
    baseline_error: str | None = None
    eager_ms: float | None = None
    achieved_gbps: float | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    warmup_iterations: int = 0
    timed_iterations: int = 0


class NsightMetrics(BaseModel):
    occupancy: float | None = None
    regs_per_thread: int | None = None
    uncoalesced_global_loads_pct: float | None = None
    spill_bytes: int = 0
    achieved_bandwidth_gbps: float | None = None
    achieved_tflops: float | None = None
    # Bottleneck signals (v1.1b) — let the perf-repair loop tell latency- from
    # bandwidth- from compute-bound instead of defaulting to "add more ILP".
    memory_throughput_pct: float | None = None
    dram_throughput_pct: float | None = None
    compute_sm_pct: float | None = None
    waves_per_sm: float | None = None
    sol_bottleneck: str = ""
    raw_csv: str = ""


class GPURunner(ABC):
    @abstractmethod
    def compile(
        self,
        src: str,
        *,
        target_arch: str,
        extra_flags: tuple[str, ...] = (),
    ) -> CompileResult:
        raise NotImplementedError

    @abstractmethod
    def run_kernel(
        self,
        so_path: Path,
        inputs: list[Any],
        timeout_seconds: int = 30,
    ) -> RunResult:
        raise NotImplementedError

    @abstractmethod
    def benchmark_kernel(
        self,
        so_path: Path,
        inputs: list[Any],
        *,
        reference: Callable[..., Any] | None = None,
        reference_path: Path | None = None,
        warmup_iterations: int = 10,
        timed_iterations: int = 50,
        timeout_seconds: int = 60,
    ) -> BenchmarkResult:
        raise NotImplementedError

    @abstractmethod
    def profile(self, so_path: Path, inputs: list[Any]) -> NsightMetrics:
        raise NotImplementedError
