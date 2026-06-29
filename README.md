# cuda-engine

> Plain English + a slow PyTorch reference → a verified, benchmarked, annotated CUDA kernel.

`cuda-engine` is a Python library and CLI that turns a natural-language description and a reference PyTorch function into a CUDA kernel that compiles, matches the reference within tolerance on a real GPU, and benchmarks against `torch.compile`. It uses Claude (Anthropic) for a 5-stage agent loop (interview → codegen → correctness → performance → polish) with Nsight-driven perf repair and Sonnet→Opus escalation when budgets bust.

**Status:** pre-1.0. Implementation is feature-complete through M3; v1.0 release gate (M4) is pending. Internal regression suite, eval runner, nightly CI, and `torch.compile` baseline measurement are all in place. See [docs/milestones/M3-evidence.md](docs/milestones/M3-evidence.md) for the most recent eval results.

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

# Run the internal eval suite (42 kernels)
cuda-engine eval --suite internal --out evals/results/2026-05-12 --resume

# Run the suite on a different provider (to benchmark models against each other)
cuda-engine eval --suite internal --out evals/results/openai --model-id openai:gpt-4o
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

The internal regression suite has 30 hand-curated kernels covering elementwise ops, reductions, and simple fused kernels. All speedups are measured on an A100 (sm_80) against the **fastest** `torch.compile` mode (best of `default` / `max-autotune` / `reduce-overhead`) at N≈16M, so a win means beating torch.compile at its best.

**Internal suite — 30/30, A100, 2026-06-01** ([M3-evidence.md](docs/milestones/M3-evidence.md)):
- Functional pass rate: **30/30 (100%)**.
- Median speedup vs torch.compile: **1.04×**; p25: **1.00×**.
- **fast_1: 24/30 (80%)** kernels strictly faster than torch.compile.
- Biggest wins: `topk_fp32` 12.5× (inductor falls back to a slow sort), `masked_mean` 2.6×, `cumulative_max` 1.45×, `softmax_lastdim` 1.33×. As expected for bandwidth-bound elementwise ops, those sit at parity (torch.compile is already at the HBM roofline); the wins come from reductions/scans.

**KernelBench external subset** (12 unseen, in-scope level1 ops): 9/9 functional on the kernels run so far (remaining 3 pending a credit top-up).

> An earlier baseline bug measured against `torch.compile`'s *slowest* mode (reduce-overhead) at too-small N, which inflated speedups (one kernel read 9.7× when the honest number is ~parity). Fixed in commit `21f3b2b`; all numbers above use the corrected best-mode baseline.

---

## Scope

### In scope for v1
- **Kernel categories:** elementwise + simple fused (RMSNorm, layernorm, GELU/SiLU/sigmoid variants, GLU/SwiGLU/GEGLU fusions, dropout-fused) and reductions/scans (sum, mean, argmax, top-k, prefix-sum, masked-mean).
- **Targets:** codegen for `sm_80` / `sm_90` / `sm_100`; runtime verification on `sm_80` only.
- **LLM:** Anthropic Claude Sonnet 4.6 default, Opus 4.7 escalation, prompt caching. **Pluggable providers (v1.1):** OpenAI and Google Gemini have native adapters, and any OpenAI-API-compatible endpoint (OpenRouter, Together, Groq, vLLM, local models) works via a generic adapter — set per-stage via `stage_models`, or run the eval on one with `--model-id`. Claude stays the default with caching + tool use; providers that lack a feature degrade gracefully and the run records it.
- **Eval suites:** 42-kernel internal regression + filtered KernelBench subset.

### Out of scope for v1
- GEMM, matmul, attention kernels (CUTLASS and FlashAttention dominate; deferred to v2/v3).
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
