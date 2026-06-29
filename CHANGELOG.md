# Changelog

All notable changes to **cuda-engine** are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-28

First public release. Turns a plain-English prompt plus a PyTorch reference
function into a verified, benchmarked, annotated CUDA kernel through a
five-stage, Claude-driven agent loop.

### Added
- **Five-stage synthesis pipeline**: Interview → Codegen → Correctness
  (hard gate) → Performance (soft gate, Nsight-guided) → Polish.
- **`cuda-engine` CLI**: kernel synthesis plus a resumable `eval` runner for the
  internal and KernelBench suites.
- **LLM backend**: Claude Sonnet 4.6 default with Opus escalation, prompt
  caching, and tool use.
- **Service interfaces**: `LLMClient`, `GPURunner`, and `ArtifactStore`, each
  with a single v1 implementation.
- **Streamlit demo** (`examples/web_demo.py`) and a Colab quickstart notebook.
- **Three worked examples**: `rmsnorm_silu_fp16`, `softmax_lastdim_fp16`,
  `topk_fp32`.
- **Evaluation suites**: 30 hand-curated internal kernels and a 12-kernel
  hand-translated KernelBench external subset (no overlap with internal).
- **Docs**: README quickstart with honest eval numbers, privacy and cost guides.

### Verified (A100, sm_80)
- **Internal suite**: 30/30 functional, median 1.04× and p25 1.00× vs the
  fastest torch.compile mode (N=16M), fast_1 24/30 (80%).
- **KernelBench external subset**: 12/12 functional, median 1.05×, p25 1.03×,
  fast_1 11/12 (92%).

### Scope
- v1 targets elementwise ops, simple fused ops, and reductions/scans.
  GEMM and attention are out of scope.

[1.0.0]: https://github.com/shivnarainms22/Cuda-Engine/releases/tag/v1.0.0
