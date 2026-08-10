from cuda_engine.api import synthesize
from cuda_engine.config import RetryBudgets, SynthesisConfig
from cuda_engine.models import (
    CorrectnessReport,
    KernelArtifact,
    KernelSpec,
    PerformanceReport,
    SynthesisReport,
    SynthesisResult,
)

__all__ = [
    "CorrectnessReport",
    "KernelArtifact",
    "KernelSpec",
    "PerformanceReport",
    "RetryBudgets",
    "SynthesisConfig",
    "SynthesisReport",
    "SynthesisResult",
    "synthesize",
]

def _read_version() -> str:
    """Read the version from distribution metadata, so it cannot drift.

    It was hardcoded to "0.0.1" through the 1.0-1.2 releases while PyPI served the
    real version. pyproject.toml is the single source of truth.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("cuda-engine")
    except PackageNotFoundError:  # running from a source tree, not installed
        return "0.0.0+unknown"


__version__ = _read_version()
