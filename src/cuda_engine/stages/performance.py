from cuda_engine.models import KernelArtifact, KernelSpec, PerformanceReport
from cuda_engine.services.store.base import ArtifactStore
from cuda_engine.stages.base import Stage
from cuda_engine.stages.correctness import _make_inputs


class Stage4Performance(Stage):
    name = "performance"

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

        inputs = _make_inputs(spec, shape=(4096,))
        benchmark = self.gpu.benchmark_kernel(
            artifact.kernel_so_path,
            inputs,
            warmup_iterations=10,
            timed_iterations=50,
        )
        self.store.write_json(run_id, "stage4_performance/benchmark.json", benchmark.model_dump(mode="json"))

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


def _write_report(store: ArtifactStore, run_id: str, report: PerformanceReport) -> None:
    store.write_json(run_id, "stage4_performance/report.json", report.model_dump(mode="json"))
