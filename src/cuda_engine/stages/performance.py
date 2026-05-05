from cuda_engine.models import KernelArtifact, KernelSpec, PerformanceReport
from cuda_engine.stages.base import Stage


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
        return PerformanceReport(speedup_vs_reference=1.0, speedup_vs_torch_compile=1.0)
