from cuda_engine.services.gpu.base import GPURunner
from cuda_engine.services.llm.base import LLMClient
from cuda_engine.services.store.base import ArtifactStore


class BudgetExhaustedError(RuntimeError):
    """Raised when a stage exhausts its retry budget without producing a valid result."""


class StructuralStageError(RuntimeError):
    """Raised when a stage cannot produce structurally valid data."""


class Stage:
    name: str = "stage"

    def __init__(
        self,
        llm: LLMClient | None = None,
        gpu: GPURunner | None = None,
        store: ArtifactStore | None = None,
    ) -> None:
        self.llm = llm
        self.gpu = gpu
        self.store = store
