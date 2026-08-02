# Changelog

All notable changes to **cuda-engine** are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`torch.compile` compatibility for generated kernels** (`cuda_engine.torch_compat`).
  Generated kernels registered a CUDA impl but no fake/meta impl, so they were
  opaque to Dynamo: `torch.compile` graph-broke on the op (splitting the compiled
  region and disqualifying it from CUDA graphs) and `torch.export`/AOTInductor
  failed outright. The shape rule is now derived mechanically from the frozen
  `KernelSpec` — no LLM involvement, so no new failure mode in the repair loop.
  - `resolve_output_shapes()` binds symbols into one table shared across *all*
    arguments, so `a:(M,K) @ b:(K,N) -> (M,N)` resolves — a case the correctness
    stage's per-argument binding cannot express.
  - Unresolvable specs raise `ShapeResolutionError` rather than guessing a shape.
  - `SymInt`-safe: sizes are never coerced with `int()`, so `dynamic=True` does
    not recompile.
  - `render_fake_module()` emits the same logic as standalone source (importing
    only `torch`) for shipping inside an exported kernel package.
  - Verified on CPU-only torch with no GPU and no API spend, including a negative
    control proving the `fullgraph=True` acceptance test can fail.

- **`cuda-engine export <run_id> --out <dir>`** — turn a verified run into a
  self-contained, `pip install`able kernel package. A run previously ended as a
  `.cu` inside a cache directory; using it meant hand-writing a
  `cpp_extension.load` call, hand-registering a fake impl, and re-deriving from
  JSON what had actually been verified.
  - The package registers the fake impl on import, so it composes inside
    `torch.compile` without a graph break.
  - Ships source and JIT-builds on first use, preferring a bundled `.so` when it
    loads — a source build is an honest fallback where a mismatched binary is a
    confusing symbol error.
  - Refuses to export a kernel whose correctness gate did not pass; `--force`
    overrides but stamps the package `UNVERIFIED`. There is no silent path to an
    unmarked unverified package.
  - `VERIFICATION.md` records the evidence *and the negative space* — which
    architecture was actually exercised, which shapes, which tolerances, which
    `torch.compile` baseline mode the speedup was measured against, and what was
    not tested at all.
- `ArtifactStore.read_bytes` — binary artifacts are now readable through the
  store interface instead of via a private attribute poke.

### Changed / hardened
- **Trustworthy perf benchmarking.** The benchmark now verifies the kernel's
  output against the reference *at the benchmark shape*, so a kernel that is fast
  but wrong at scale can no longer post a speedup; and the correctness hard gate
  verifies rank-≥2 kernels at the benchmark shape so scale bugs are repaired, not
  hidden. Baseline benchmark timeout is now configurable (fixes `torch.compile`
  GEMM baselines timing out).

### In progress
- GEMM/matmul (v2.0) — eval suite + measurement/correctness integrity fixes are in.
  Validated win: `matmul_bias_gelu_fp16` at **1.25×** vs torch's fused path (real,
  correct-at-4096², CUDA-core fused epilogue). A tensor-core (WMMA) codegen-guidance
  experiment (Rung 4) was tried and **reverted**: it fixed the `matmul_fp16` compile
  failure but steered the fused kernel onto naive WMMA (~10% peak), collapsing the
  1.25× win to 0.09×. Bare fp16 GEMM vs cuBLAS is not the goal; the fused CUDA-core
  epilogue is. See `docs/milestones/v2.0-gemm-rung4-evidence.md`.

## [1.2.0] - 2026-07-08

Backward-compatible. Robustness + observability.

### Added
- **Synthesis-stage resumability** — `cuda-engine synthesize --resume <run_id>`
  (and `synthesize(..., resume_run_id=...)`) resumes a run killed mid-pipeline
  (Colab disconnect, credit exhaustion), reusing completed stages from a
  `checkpoint.json` with an inputs-fingerprint guard so it never resumes against
  a changed prompt/reference.
- **Cross-provider comparison** — `cuda-engine compare-providers <run>… --out cmp.md`
  combines per-provider eval results into a "which model writes the best CUDA"
  table (functional %, median/p25 speedup, fast_1 per model). Run the suite once
  per provider (`eval --model-id …`), then combine.

### Fixed
- Eval failure classification: a codegen budget-exhaustion / structural stage
  error is now reported as a `stage_failure` (the engine cleanly gave up inside a
  stage), not `runner_error` (which wrongly implied infra flakiness).

[1.2.0]: https://github.com/shivnarainms22/Cuda-Engine/releases/tag/v1.2.0

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
