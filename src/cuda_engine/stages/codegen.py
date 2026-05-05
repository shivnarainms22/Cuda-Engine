from pathlib import Path

from cuda_engine.models import KernelArtifact, KernelSpec
from cuda_engine.stages.base import Stage


class Stage2Codegen(Stage):
    name = "codegen"

    def run(self, *, spec: KernelSpec, run_id: str, retry_budget: int = 3) -> KernelArtifact:
        return KernelArtifact(kernel_cu_path=Path("/tmp/stub.cu"))
