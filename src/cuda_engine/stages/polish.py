from cuda_engine.models import CorrectnessReport, KernelArtifact, KernelSpec, PerformanceReport
from cuda_engine.stages.base import Stage


class Stage5Polish(Stage):
    name = "polish"

    def run(
        self,
        *,
        spec: KernelSpec,
        artifact: KernelArtifact,
        correctness: CorrectnessReport,
        performance: PerformanceReport,
        run_id: str,
    ) -> KernelArtifact:
        return artifact
