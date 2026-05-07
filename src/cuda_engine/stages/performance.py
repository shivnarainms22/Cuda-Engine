from typing import Any

from cuda_engine.config import SynthesisConfig
from cuda_engine.models import KernelArtifact, KernelSpec, PerformanceReport
from cuda_engine.services.gpu.base import BenchmarkResult, GPURunner
from cuda_engine.services.llm.base import LLMClient
from cuda_engine.services.store.base import ArtifactStore
from cuda_engine.stages.base import Stage
from cuda_engine.stages.correctness import _make_inputs


class Stage4Performance(Stage):
    name = "performance"

    def __init__(
        self,
        llm: LLMClient | None = None,
        gpu: GPURunner | None = None,
        store: ArtifactStore | None = None,
        cfg: SynthesisConfig | None = None,
    ) -> None:
        super().__init__(llm=llm, gpu=gpu, store=store)
        self.cfg = cfg or SynthesisConfig()

    def run(
        self,
        *,
        spec: KernelSpec,
        artifact: KernelArtifact,
        run_id: str,
        retry_budget: int = 3,
    ) -> PerformanceReport:
        if self.gpu is None or self.store is None:
            raise RuntimeError("Stage4Performance requires gpu and store services")
        if artifact.kernel_so_path is None:
            report = PerformanceReport(
                speedup_vs_reference=0.0,
                speedup_vs_torch_compile=0.0,
                below_target=True,
                notes=["kernel_so_path is required for performance benchmarking"],
            )
            _write_report(self.store, run_id, report)
            return report

        benchmark_shape = _benchmark_shape(spec, total_elements=self.cfg.performance_shape_n)
        inputs = _make_inputs(spec, shape=benchmark_shape)
        benchmark = self.gpu.benchmark_kernel(
            artifact.kernel_so_path,
            inputs,
            warmup_iterations=self.cfg.benchmark_warmup_iterations,
            timed_iterations=self.cfg.benchmark_timed_iterations,
        )
        self.store.write_json(
            run_id,
            "stage4_performance/benchmark.json",
            _benchmark_payload(benchmark, cfg=self.cfg),
        )

        if not benchmark.ok:
            report = PerformanceReport(
                speedup_vs_reference=0.0,
                speedup_vs_torch_compile=0.0,
                below_target=True,
                notes=[benchmark.stderr or "benchmark failed"],
            )
            _write_report(self.store, run_id, report)
            return report

        speedup = _speedup(baseline_ms=benchmark.baseline_ms, custom_ms=benchmark.custom_ms)
        report = PerformanceReport(
            speedup_vs_reference=speedup,
            speedup_vs_torch_compile=speedup,
            achieved_gbps=benchmark.achieved_gbps,
            below_target=speedup < 1.0,
        )
        _write_report(self.store, run_id, report)
        return report


def _speedup(*, baseline_ms: float | None, custom_ms: float) -> float:
    if baseline_ms is None or custom_ms <= 0:
        return 1.0
    return baseline_ms / custom_ms


def _benchmark_shape(spec: KernelSpec, *, total_elements: int) -> tuple[int, ...]:
    rank = max((len(arg.shape) for arg in spec.inputs), default=1)
    if rank <= 1:
        return (total_elements,)
    dim = max(1, round(total_elements ** (1 / rank)))
    return tuple(dim for _ in range(rank))


def _write_report(store: ArtifactStore, run_id: str, report: PerformanceReport) -> None:
    store.write_json(run_id, "stage4_performance/report.json", report.model_dump(mode="json"))


def _benchmark_payload(benchmark: BenchmarkResult, *, cfg: SynthesisConfig) -> dict[str, Any]:
    payload = benchmark.model_dump(mode="json")
    payload["settings"] = {
        "performance_shape_n": cfg.performance_shape_n,
        "benchmark_warmup_iterations": cfg.benchmark_warmup_iterations,
        "benchmark_timed_iterations": cfg.benchmark_timed_iterations,
    }
    return payload
