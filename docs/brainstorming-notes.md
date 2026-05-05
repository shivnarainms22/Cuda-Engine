# CUDA Synthesis Engine — Brainstorming Notes

**Status:** ✅ **Brainstorming complete (2026-04-26).** Final design at `docs/superpowers/specs/2026-04-26-cuda-synthesis-engine-design.md`.
**Started:** 2026-04-26
**Source spec:** `Pro_CUDA_Master_Spec_2026.pdf` (Pro-CUDA AI Synthesis Engine, v1.0 April 2026)

> Running log of decisions, suggestions, and concerns from the brainstorming phase.
> Frozen now that brainstorming is complete. Future changes go through the design doc + plan, not this file.

---

## Project Vision (from user)

A tool that takes a plain-English prompt and produces high-quality CUDA kernels for the latest NVIDIA hardware (Blackwell / Rubin), targeting senior-engineer quality output and following modern industry standards. End goal: a real product. Open-source is acceptable — quality bar must remain high.

---

## All Decisions Locked

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Project type | **Open-source quality product** (not paid) | User priority is quality and broad usefulness. |
| 2 | Target users | **A + B + C + D** (ML researchers, perf engineers, inference platforms, migration consultants) | Broad ML/infra audience; ruled out E (graphics/HPC). |
| 3 | Hardware strategy | **A100 first → Blackwell later** when cloud B200 hours are accessible | Only A100 + Colab Pro available now; can't credibly ship Blackwell-only without testing on Blackwell. |
| 4 | Kernel categories for v1 | **A. Elementwise + simple fused ops** + **B. Reductions / scans** | Wide utility, proves the loop without fighting CUTLASS. |
| 5 | Input UX | **English prompt + PyTorch reference function + target arch flag** | The reference IS the oracle. Verifiable correctness as a first-class feature. |
| 6 | Agentic loop shape | **5-stage agent loop with bounded retries per stage** | Observable, debuggable, swappable per-stage models. |
| 7 | Eval suite | **KernelBench subset (external) + ~30 hand-curated kernels (internal)** | Public credibility + fast dev feedback. |
| 8 | Performance bar (v1 ship) | **Pragmatic: ≥95% functional internal, ≥80% KernelBench, median ≥1.0× torch.compile, p25 ≥0.7×, ≥30% fast_1** | Defensible numbers, achievable in v1. |
| 9 | LLM backend | **Claude only (Sonnet 4.6 default, Opus 4.7 escalation), prompt caching, tool use** | Single dependency, ship one path well. Pluggable later. |
| 10 | Interface | **Python library + thin CLI + bundled Streamlit demo** | Library is canonical; CLI for CI; demo for first-five-minutes UX. |
| 11 | Architecture | **Monolithic library, single process, with clean service interfaces** | Simplest v1; service interfaces leave the door open to remote-runner v2. |

---

## Loop Architecture

```
Stage 1: Architectural Interview      Resolve ambiguity from prompt + reference.
                                      Output: frozen KernelSpec.

Stage 2: Codegen                      Generate kernel.cu + wrapper.
                                      Compile; on error, retry (≤3).

Stage 3: Correctness (HARD GATE)      Run vs PyTorch reference on shape grid.
                                      On mismatch, retry (≤3). On final fail: do not ship.

Stage 4: Performance (soft gate)      Benchmark vs torch.compile.
                                      If under target, profile with Nsight, retry (≤3).
                                      Below bar: ship with warning.

Stage 5: Polish                       Annotate kernel + emit synthesis report.
```

Sonnet→Opus escalation: each stage starts with Sonnet 4.6; on retry budget exhaustion, one final attempt with Opus 4.7. Logged in run report.

---

## Components Locked

- **Public API:** `synthesize(prompt, reference, target) → SynthesisResult`
- **Orchestrator:** owns retry budgets, stage sequencing, escalation
- **5 Stage classes:** `Stage1Interview`, `Stage2Codegen`, `Stage3Correctness`, `Stage4Performance`, `Stage5Polish`
- **Service interfaces (1 impl each in v1):**
  - `LLMClient` → `AnthropicClient` (prompt cache, tool use)
  - `GPURunner` → `LocalGPURunner` (subprocess nvcc/run/Nsight)
  - `ArtifactStore` → `LocalDirStore` (flat-file run dirs)
- **Output artifact:** `torch.utils.cpp_extension.load_inline`-loaded callable + run report

Tools registered for Claude:
- `compile_kernel(src) → {ok, errors[], warnings[], ptx_size}`
- `run_correctness(kernel_so, ref_fn, shapes) → diff stats`
- `nsight_profile(kernel_so, sample_input) → metrics`

---

## Suggestions Given (now folded into the design)

### Concerns about original PDF spec
- **Technical inaccuracies.** Blackwell is `sm_100`/`sm_120`; Hopper is `sm_90`. The PDF swapped them. "ProofWright SMT" doesn't appear to be a real tool.
- **Drop "1,200x dev cycle speedup"** — unfalsifiable marketing.
- **Multi-model pipeline overkill** — earn the right to multi-model after one provider stops being enough.
- **Formal verification (race-freedom proofs) is research, not v1.** Defer indefinitely.
- **BlueField-4 / NVL72 / DPU offload / rack-scale orchestration** is not v1 scope. Defer to v3+.

### Wedge / value framing
- **Verification is the wedge.** "Every kernel is numerically verified vs reference + benchmarked + reported."
- **Architecture migration** is a strong secondary wedge for later phases.
- **Long-tail custom kernels** (fused ops, novel attention variants) is the realest customer pain.

### Three-ring scoping
- **Ring 1 (build now):** Synthesis → compile → numerical check → benchmark → retry loop, on one kernel family, with a real eval suite. ← decisions above land us here.
- **Ring 2 (next):** Multi-kernel coverage, autotuning, real Nsight-driven feedback, multi-arch.
- **Ring 3 (research):** Formal proofs, multi-arch cost modeling, rack-scale orchestration.

---

## Hardware & Tooling Realities (constraints baked into design)

- **A100 (sm_80, Ampere):** 3rd-gen tensor cores. BF16/FP16/TF32/INT8. **No FP8** (Hopper+). **No NVFP4/TMEM** (Blackwell+).
- **CUTLASS 3.x**, not 4.x.
- **Colab Pro caveats:** session disconnects, ephemeral filesystem, A100 not always available.
- **Cloud B200 hourly** approachable in 2026 (~$5–8/hr) for milestone-style verification once v1 works.

---

## Architecture Principle

Target architecture is a first-class flag (`--target sm_80 | sm_90 | sm_100`). Arch-specific knowledge lives in pluggable modules (`targets/sm_80.py`, etc.). Verification runs on whatever silicon is available; codegen targets whatever the user requests.

---

## Changelog

- **2026-04-26 09:00** — Initial notes file created. Decisions 1–6 captured.
- **2026-04-26 14:30** — Design Sections 1–6 approved. Decisions 7–11 added. Brainstorming closed. See spec doc.
