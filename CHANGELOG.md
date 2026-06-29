# Changelog

All notable changes to **cuda-engine** are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-06-29

"Sharpen" — pluggable LLM providers, a bound-aware performance-repair loop, and
broader eval coverage. A100, fully backward-compatible (Anthropic stays the
default).

### Added
- **Pluggable LLM providers.** Native adapters for OpenAI and Google Gemini, a
  generic adapter for any OpenAI-API-compatible endpoint (OpenRouter, Together,
  Groq, vLLM, local models), and an `LLMRouter` that dispatches on namespaced
  model ids. Per-stage routing via `SynthesisConfig.stage_models`; run the eval
  on any provider with `cuda-engine eval --model-id <provider:model>`. Claude
  keeps prompt caching + tool use; providers lacking a feature degrade
  gracefully and the run records it.
- **Bound-aware Stage-4 perf repair.** `parse_ncu_csv` now surfaces the
  bottleneck signals (memory/DRAM/compute %, waves per SM, the SOL verdict) it
  previously discarded; the repair prompt classifies latency- vs bandwidth- vs
  compute-bound and gives targeted guidance instead of always adding ILP.
- **+12 in-scope internal eval kernels** (internal suite 30 → 42).
- `provider` and `model_id` columns in eval `results.csv`.

### Fixed
- `gelu_fp16` 0.67× → **1.08×** and `rms_norm_fp16` 0.75× → **1.08×** vs
  torch.compile (A100), via the bound-aware repair — both were latency/occupancy-
  bound, not roofline-bound.
- Default `synthesize()` now builds the provider router so per-stage model ids
  resolve correctly.

### Deferred to v1.2
- Automated cross-provider comparison report (benchmarking works today by
  running the eval per `--model-id` and diffing summaries).

[1.1.0]: https://github.com/shivnarainms22/Cuda-Engine/releases/tag/v1.1.0

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
