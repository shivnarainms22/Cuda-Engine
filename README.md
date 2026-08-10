# cuda-engine

> Plain English + a PyTorch reference → a verified, benchmarked CUDA kernel you can `pip install` and `torch.compile`.

`cuda-engine` turns a natural-language description and a reference PyTorch function into a CUDA kernel that compiles, matches the reference within tolerance on a real GPU, and beats `torch.compile` at its best mode — then packages it as an installable Python module that composes inside a compiled graph.

It uses a 5-stage LLM agent loop (interview → codegen → correctness → performance → polish) with Nsight-driven perf repair. Claude is the default; OpenAI, Gemini, and any OpenAI-compatible endpoint work too.

**The part most kernel-generation tools skip: proving the number is real.** LLMs are very good at producing kernels that look fast and are wrong. This project's measurement harness is built to catch its own false positives, and has repeatedly done so — see [Why you can trust the numbers](#why-you-can-trust-the-numbers).

```bash
pip install cuda-engine
cuda-engine synthesize --prompt "fp16 RMSNorm over the last dimension" --reference rms_norm.py
cuda-engine export <run_id> --out ./my_kernel   # installable, torch.compile-ready
```

---

## What it does

```python
import torch
from cuda_engine import synthesize

def rms_norm(x):
    return x * (x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-5).rsqrt().to(x.dtype)

result = synthesize(
    prompt="Generate a fp16 RMSNorm kernel without gamma over the last dimension.",
    reference=rms_norm,
    target="sm_80",
)

assert result.passed
assert result.correctness.passed                            # verified vs the reference
assert result.performance.below_target is False             # ≥1.0× torch.compile
print(f"Speedup: {result.performance.speedup_vs_torch_compile:.2f}x")
print(f"Kernel: {result.artifacts_dir}/stage5_polish/final/kernel.cu")
```

Each `synthesize()` call produces a run directory under `~/.cache/cuda_engine/runs/<run_id>/` containing every prompt sent, every LLM response, every kernel attempt, the final kernel source, the compiled shared object, and the full synthesis trace.

---

## Quickstart

### Install

Requires Python 3.11+, CUDA 12.x toolchain (`nvcc`), PyTorch 2.4+, and an A100-class GPU for end-to-end runs.

```bash
pip install cuda-engine    # post-v1.0 release
# or, from source:
git clone https://github.com/shivnarainms22/Cuda-Engine.git
cd Cuda-Engine
pip install -e ".[dev]"
```

Set your Anthropic key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### CLI

```bash
# Synthesize a single kernel
cuda-engine synthesize \
    --prompt "Generate a fp16 RMSNorm kernel without gamma over the last dimension." \
    --reference path/to/rms_norm.py \
    --target sm_80

# Inspect a previous run
cuda-engine inspect <run_id>

# Resume a run that died mid-pipeline (Colab disconnect, credit exhaustion, ...)
# — completed stages are reused from disk, not re-paid. Same prompt/reference required.
cuda-engine synthesize --resume <run_id> \
    --prompt "..." --reference path/to/rms_norm.py --target sm_80

# Run the internal eval suite (42 kernels)
cuda-engine eval --suite internal --out evals/results/2026-05-12 --resume

# Run the suite on a different provider (to benchmark models against each other)
cuda-engine eval --suite internal --out evals/results/openai --model-id openai:gpt-4o

# Compare providers: which model writes the best CUDA? (combines prior runs, no cost)
cuda-engine compare-providers evals/results/anthropic evals/results/openai --out compare.md

# Export a verified run as an installable, torch.compile-ready package
cuda-engine export <run_id> --out ./my_kernel
```

`path/to/rms_norm.py` should define either a top-level `REFERENCE` variable or a top-level `reference()` function.

### Library

```python
from cuda_engine import SynthesisConfig, synthesize
from cuda_engine.config import RetryBudgets

result = synthesize(
    prompt="...",
    reference=my_pytorch_fn,
    target="sm_80",
    config=SynthesisConfig(
        retry_budgets=RetryBudgets(codegen=3, performance=2),
        escalate_to_opus_on_bust=True,
        perf_target_speedup_vs_torch_compile=1.0,
    ),
)
```

See [`docs/cost.md`](docs/cost.md) for tuning retry budgets to bound API spend.

**Choosing a provider per stage** (v1.1) — `stage_models` maps each of the five
stages to a `"provider:model"` id (default: all `anthropic:claude-sonnet-4-6`):

```python
from cuda_engine.config import StageModels, SynthesisConfig

config = SynthesisConfig(
    stage_models=StageModels(
        interview="openai:gpt-4o",          # cheap stage on a cheap model
        codegen="anthropic:claude-sonnet-4-6",
        performance="anthropic:claude-opus-4-7",
        # ... correctness / polish
    )
)
```

Set the matching key in the environment (`OPENAI_API_KEY`, `GEMINI_API_KEY`,
or your OpenAI-compatible endpoint's key). A bare model id with no `provider:`
prefix routes to Anthropic.

---

## Shipping the kernel

A run directory is a result. `export` turns it into a dependency:

```bash
cuda-engine export <run_id> --out ./my_kernel
pip install ./my_kernel
```

```python
import torch
from ce_rms_norm_fp16 import forward

out = forward(x)

# Composes inside a compiled graph -- no graph break:
compiled = torch.compile(lambda t: forward(t) * 2, fullgraph=True)
```

The package registers a **fake (meta) implementation** derived mechanically from the frozen `KernelSpec`, which is what makes `torch.compile`, `torch.export`, and AOTInductor work. A custom CUDA op without one is opaque to Dynamo: it graph-breaks, splitting the compiled region and disqualifying it from CUDA graphs.

It ships the kernel source and JIT-builds on first use (preferring a bundled `.so` when it loads), plus:

- `VERIFICATION.md` — what was verified, and **what was not**
- `spec.json` — the frozen input/output contract
- `manifest.json` — which kernel shipped (polished or fallback), run id, provenance

`export` **refuses** a run whose correctness gate did not pass. `--force` overrides it but stamps the package `UNVERIFIED`; there is no silent path to an unmarked unverified package.

**This loop is hardware-validated**, not asserted: on an A100, an exported package builds a wheel, installs, imports in a process where `cuda_engine` is absent, JIT-builds its kernel, matches the PyTorch reference, traces under `torch.compile(fullgraph=True)`, and — measured on the installed artifact, not inherited from the run — is correct at the benchmark shape and still within 1.25× of the kernel time its original run recorded. A control proves the correctness comparison can fail. Evidence, and the five defects that validation exposed (including one in the checker itself), are in [v2.2-export-evidence.md](docs/milestones/v2.2-export-evidence.md). Reproduce it with `tools/export_validation/validate_export.py` — costs no API credits.

---

## Why you can trust the numbers

Most published kernel-generation results are single speedup figures with no way to check them. Speedups here are built to survive scrutiny, because the harness is designed to fail loudly:

- **Correctness is a hard gate.** Outputs are compared elementwise against the PyTorch reference at multiple shapes — including awkward ones (0, 1, 127, 4097) — on real hardware. A kernel that misses tolerance fails outright.
- **The baseline is torch.compile at its *best*.** The fastest of `default` / `max-autotune-no-cudagraphs` / `reduce-overhead`, not the first mode tried. Beating a deliberately weak baseline is easy, so the harness refuses to use one.
- **Correctness is re-verified at the benchmark shape.** Kernels frequently pass at 1024² and break at 4096² (index overflow, tiling edges). A kernel that is wrong at the shape it was timed at cannot post a speedup.
- **Every export states its negative space.** `VERIFICATION.md` lists the single architecture actually exercised, the exact shapes tested, the tolerances, and what was never checked — other shapes, non-contiguous layouts, streams, backward. A document that only lists successes is marketing.

This is not theoretical. The harness has caught its own false positives:

| What was claimed | What was true | How it was caught |
|---|---|---|
| `sigmoid_mul` 9.7× | ~parity | Baseline was `torch.compile`'s *slowest* mode at too-small N ([`21f3b2b`](https://github.com/shivnarainms22/Cuda-Engine/commit/21f3b2b)) |
| `matmul_bias_gelu` 1.11×, `matmul_fp32` 0.91× | both wrong at the benchmark shape | Correctness-at-benchmark-shape gate; both were correct at ≤1024² and garbage at 4096² |
| WMMA codegen guidance would help fp16 GEMM | traded a real 1.25× win for a pass on an unwinnable kernel | Measured, found to be a net regression, and reverted |

Published numbers below are post-fix and honest, including the ones that lost.

---

## How it works

```
   prompt + reference.py
            │
            ▼
   ┌─────────────────────┐
   │  Stage 1: Interview │ → KernelSpec (frozen contract)
   └─────────────────────┘
            │
            ▼
   ┌─────────────────────┐
   │  Stage 2: Codegen   │ → kernel.cu + compile.log (hard retry budget)
   └─────────────────────┘
            │
            ▼
   ┌─────────────────────┐    fail → repair via Stage 2
   │  Stage 3: Correct.  │ ──────────┐
   │  HARD GATE          │           │
   └─────────────────────┘           │
            │ pass                   ▼
            ▼                  (loop until pass or budget exhausted)
   ┌─────────────────────┐
   │  Stage 4: Perf      │ → benchmark vs torch.compile
   │  SOFT GATE          │    Nsight-driven repair loop
   │                     │    Sonnet → Opus escalation
   └─────────────────────┘
            │
            ▼
   ┌─────────────────────┐
   │  Stage 5: Polish    │ → annotated kernel.cu (re-verified)
   └─────────────────────┘
            │
            ▼
   SynthesisResult + run_dir
```

- **Hard gate (Stage 3):** kernels that don't match the reference within tolerance fail outright. No exceptions.
- **Soft gate (Stage 4):** kernels below the perf target still ship, but with `below_target=True` and a warning. Stage 4 burns its retry budget on Nsight-driven optimizations, then optionally escalates to Opus.
- **Subprocess isolation:** all GPU work happens in a subprocess child. Crashes in user kernels (segfaults, illegal memory access, OOM) don't take down the orchestrator.

Design document: [`docs/superpowers/specs/2026-04-26-cuda-synthesis-engine-design.md`](docs/superpowers/specs/2026-04-26-cuda-synthesis-engine-design.md).

---

## Eval results

The internal regression suite has **42** hand-curated kernels covering elementwise ops, reductions, and simple fused kernels. All speedups are measured on an A100 (sm_80) against the **fastest** `torch.compile` mode (best of `default` / `max-autotune` / `reduce-overhead`) at N≈16M, so a win means beating torch.compile at its best.

**v1.0 gate — A100** ([M3-evidence.md](docs/milestones/M3-evidence.md), [M4-evidence.md](docs/milestones/M4-evidence.md)):
- Internal suite (30 at release): **30/30 (100%)**, median **1.04×**, p25 **1.00×**, fast_1 **24/30 (80%)**.
- KernelBench external subset: **12/12 (100%)**, median **1.05×**.
- Biggest wins: `topk_fp32` 12.5× (inductor falls back to a slow sort), `masked_mean` 2.6×, `cumulative_max` 1.45×, `softmax_lastdim` 1.33×. Bandwidth-bound elementwise ops sit at parity (torch.compile is already at the HBM roofline); the wins come from reductions/scans.

v1.1 added 12 more in-scope kernels (suite → 42) and the ability to benchmark providers against each other (`compare-providers`); v1.2 added resumability. GEMM/matmul (v2.0) is merged on `main` but out of this in-scope suite — its status is a separate track: `matmul_bias_gelu_fp16` **1.25×** (real fused-epilogue win vs torch's fused path, correct at 4096²), `matmul_fp32` 0.66×, and bare `matmul_fp16` is deliberately not pursued (naive tensor-core GEMM can't beat cuBLAS, and wasn't the goal). See [v2.0-gemm-rung4-evidence.md](docs/milestones/v2.0-gemm-rung4-evidence.md).

**KernelBench external subset** (12 unseen, in-scope level1 ops): 12/12 functional, hand-translated with no overlap with the internal suite.

> An earlier baseline bug measured against `torch.compile`'s *slowest* mode (reduce-overhead) at too-small N, which inflated speedups (one kernel read 9.7× when the honest number is ~parity). Fixed in commit `21f3b2b`; all numbers above use the corrected best-mode baseline.

---

## Status

**v1.2 released** ([PyPI](https://pypi.org/project/cuda-engine/)). v1.0 shipped the full 5-stage loop (A100-verified); v1.1 added pluggable LLM providers and a bound-aware perf-repair loop; v1.2 added synthesis-stage resumability and cross-provider comparison.

Merged on `main`, not yet in a PyPI release:

- **GEMM (v2.0).** The fused-epilogue thesis holds: `matmul_bias_gelu_fp16` at **1.25×** vs torch's fused path, correct at 4096². Bare fp16 GEMM vs cuBLAS is explicitly *not* pursued — naive tensor-core GEMM lands around 10% of peak and that was never the goal.
- **`torch.compile` compatibility + `export`** — the deployability work described above, validated end to end on an A100.

### Honest limits

- Runtime verification is **sm_80 (A100) only**. `sm_90`/`sm_100` are codegen targets that have never been executed. If you are on Blackwell, treat this as unverified.
- Synthesis costs API tokens (~$0.10–2.00 per kernel) and needs a GPU with `nvcc`.
- Bandwidth-bound elementwise ops sit at parity — `torch.compile` is already at the HBM roofline there, so ~1.0× is the physical ceiling, not a defect. Real wins come from reductions, scans, and fusions.
- Forward pass only. No autograd formulas are generated.

---

## Scope

### In scope for v1
- **Kernel categories:** elementwise + simple fused (RMSNorm, layernorm, GELU/SiLU/sigmoid variants, GLU/SwiGLU/GEGLU fusions, dropout-fused) and reductions/scans (sum, mean, argmax, top-k, prefix-sum, masked-mean).
- **Targets:** codegen for `sm_80` / `sm_90` / `sm_100`; runtime verification on `sm_80` only.
- **LLM:** Anthropic Claude Sonnet 4.6 default, Opus 4.7 escalation, prompt caching. **Pluggable providers (v1.1):** OpenAI and Google Gemini have native adapters, and any OpenAI-API-compatible endpoint (OpenRouter, Together, Groq, vLLM, local models) works via a generic adapter — set per-stage via `stage_models`, or run the eval on one with `--model-id`. Claude stays the default with caching + tool use; providers that lack a feature degrade gracefully and the run records it.
- **Eval suites:** 42-kernel internal regression + filtered KernelBench subset.

### Out of scope for v1
- GEMM, matmul, attention kernels (CUTLASS and FlashAttention dominate). GEMM is now being explored in v2.0 on `main` — fused epilogues win (1.25×), bare GEMM vs cuBLAS is not pursued; attention stays deferred to v3.
- Multi-GPU, multi-node, rack-scale orchestration.
- Formal verification (SMT race-freedom proofs).
- Backward-pass kernel synthesis, autograd custom ops.
- VS Code / IDE integrations.

---

## Cost

Per-kernel envelope under default config:

| Scenario | USD |
|---|---|
| Happy path | ~$0.10–0.20 |
| Typical with retries | ~$0.15–0.40 |
| Hard kernel | ~$0.30–0.80 |
| With Opus escalation | ~$0.80–2.00 |

Full eval suite (30 kernels): ~$5–20 depending on retries. See [`docs/cost.md`](docs/cost.md) for the per-stage breakdown and the four config knobs to bound spend.

---

## Privacy

`cuda-engine` writes full LLM transcripts and reference source code to `~/.cache/cuda_engine/runs/<run_id>/`. No telemetry, no third-party logging. All network traffic is to `api.anthropic.com` over TLS. See [`docs/privacy.md`](docs/privacy.md) for how to keep proprietary references out of artifact directories.

---

## Examples

- [`examples/notebook.ipynb`](examples/notebook.ipynb) — Colab quickstart (5 cells).
- [`examples/web_demo.py`](examples/web_demo.py) — Streamlit live demo.
- [`examples/kernels/`](examples/kernels/) — worked examples with prompt, reference, generated kernel, and synthesis report.

---

## Development

```bash
pip install -e ".[dev]"
ruff check src tests evals
mypy src
pytest tests/unit -v
pytest tests/integration -v -m integration   # requires CUDA + ANTHROPIC_API_KEY
```

CI:
- **PR workflow** ([.github/workflows/pr.yml](.github/workflows/pr.yml)) — unit tests, ruff, mypy on every push/PR.
- **Nightly workflow** ([.github/workflows/nightly.yml](.github/workflows/nightly.yml)) — full integration suite + eval on self-hosted A100, daily cron.
- **Pre-release workflow** ([.github/workflows/eval.yml](.github/workflows/eval.yml)) — manual trigger, gates v1.0 release.

---

## License

MIT. See [`LICENSE`](LICENSE).

---

## Acknowledgements

Built on top of Anthropic's Claude API, PyTorch's `torch.utils.cpp_extension`, NVIDIA's CUDA toolkit and Nsight Compute. Internal regression kernels draw inspiration from [KernelBench](https://github.com/ScalingIntelligence/KernelBench).
