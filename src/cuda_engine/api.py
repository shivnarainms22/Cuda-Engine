from collections.abc import Callable
from typing import Any

from cuda_engine.config import SynthesisConfig
from cuda_engine.models import SynthesisResult
from cuda_engine.orchestrator import Orchestrator
from cuda_engine.services.gpu.base import GPURunner
from cuda_engine.services.llm.base import LLMClient
from cuda_engine.services.store.base import ArtifactStore


def synthesize(
    prompt: str,
    reference: Callable[..., Any],
    target: str = "sm_80",
    config: SynthesisConfig | None = None,
    *,
    _llm: LLMClient | None = None,
    _gpu: GPURunner | None = None,
    _store: ArtifactStore | None = None,
) -> SynthesisResult:
    """Synthesize a CUDA kernel from English prompt + PyTorch reference."""

    cfg = config or SynthesisConfig()
    if _llm is None:
        from cuda_engine.services.llm.anthropic import AnthropicClient

        _llm = AnthropicClient(cfg=cfg)
    if _gpu is None:
        from cuda_engine.services.gpu.local import LocalGPURunner

        _gpu = LocalGPURunner(cfg=cfg)
    if _store is None:
        from cuda_engine.services.store.local_dir import LocalDirStore

        _store = LocalDirStore(cfg=cfg)

    orchestrator = Orchestrator(llm=_llm, gpu=_gpu, store=_store, cfg=cfg)
    return orchestrator.run(prompt=prompt, reference=reference, target=target)
