from dataclasses import dataclass

from cuda_engine.services.gpu.base import GPURunner
from cuda_engine.services.llm.base import LLMClient
from cuda_engine.services.store.base import ArtifactStore


@dataclass(frozen=True)
class SonnetFailureSummary:
    """Captures Sonnet's final failed-attempt state for handoff to Opus."""

    last_compile_errors: str
    last_compile_log: str
    last_source_attempt: str
    attempts_made: int


class BudgetExhaustedError(RuntimeError):
    """Raised when a stage exhausts its retry budget without producing a valid result."""

    def __init__(self, message: str, summary: SonnetFailureSummary | None = None) -> None:
        super().__init__(message)
        self.summary = summary


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
