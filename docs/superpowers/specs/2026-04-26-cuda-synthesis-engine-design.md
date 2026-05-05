# CUDA Synthesis Engine — Design Document (v1)

| Field | Value |
|---|---|
| **Status** | Approved (2026-04-26) |
| **Version** | 1.0 |
| **Owner** | Shivnarain |
| **Source brainstorm** | `docs/brainstorming-notes.md` |
| **Implementation plan** | `docs/superpowers/plans/2026-04-26-cuda-synthesis-engine-plan.md` (to be created) |

---

## 1. Vision

A tool that turns a plain-English prompt + a slow-but-correct PyTorch reference function into a high-performance, **verified** CUDA kernel — with senior-engineer code quality, full provenance, and a numerical-correctness guarantee. Open-source, quality-first, primary persona = ML researchers, perf engineers, inference-platform builders, and migration consultants.

---

## 2. Scope

### In scope for v1
- **Kernel categories:** elementwise + simple fused ops (RMSNorm, fused activation+bias, softmax, layernorm, GLU/GELU/SiLU fusions, dropout-fused variants) and reductions/scans (sum, mean, argmax, top-k, segment-sum, prefix-sum, masked-mean).
- **Target architectures (codegen):** `sm_80` (A100), `sm_90` (H100), `sm_100` (Blackwell B200). Codegen-aware for all three.
- **Verification (real silicon):** `sm_80` only in v1. `sm_90`/`sm_100` are codegen-only until cloud GPU hours are budgeted.
- **Input UX:** English prompt + PyTorch reference function + target arch flag.
- **Output:** loadable Torch op (via `torch.utils.cpp_extension.load_inline`) + annotated kernel source + synthesis report.
- **Interface:** Python library + thin CLI + bundled Streamlit demo.
- **LLM backend:** Anthropic only (Claude Sonnet 4.6 default, Opus 4.7 escalation), prompt caching, tool use.
- **Eval:** ~30 hand-curated kernels (internal regression) + filtered KernelBench subset (external).

### Out of scope for v1
- GEMM / matmul / attention kernels (defer to v2/v3 — CUTLASS and FlashAttention dominate).
- Multi-GPU, multi-node, rack-scale orchestration.
- Formal verification (SMT race-freedom proofs).
- Cross-LLM-provider support (Anthropic-only behind a single seam).
- BlueField/DPU offload, NVLink-aware sharding, MoE expert routing.
- Backward-pass kernel synthesis, autograd-aware custom ops.
- Persistent run database, run resumability.
- VS Code / IDE integrations.

### Explicitly rejected from the original PDF spec
- "1,200× development speedup" claim — unfalsifiable, dropped.
- Multi-model pipeline (Claude + Nemotron + GPT) — earn the right to multi-model later.
- Formal verification ("ProofWright SMT") — research project, not a product feature.
- Rack-scale infra (NVL72, BlueField-4, DOCA Memos) — not buildable on A100.

---

## 3. Architecture

### 3.1 Component map (three layers)

```
┌──────────────────────────────────────────────────────────────────────┐
│  Public API:   cuda_engine.synthesize(prompt, reference, target)     │
│                cuda-engine CLI                                       │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Orchestrator + Stages   (pure logic, no I/O)                        │
│  - Orchestrator: drives 5 stages, owns retry budgets + escalation    │
│  - Stages: Interview / Codegen / Correctness / Performance / Polish  │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Services   (interfaces, side effects isolated)                      │
│  - LLMClient        → AnthropicClient                                │
│  - GPURunner        → LocalGPURunner                                 │
│  - ArtifactStore    → LocalDirStore                                  │
└──────────────────────────────────────────────────────────────────────┘
```

**Layer rule:** stages never touch network or filesystem directly; they call services through interfaces. Each interface has exactly one v1 implementation, but the boundary exists from day 1 (cheap insurance against rewrite when v2 needs remote runners or alternate LLM providers).

### 3.2 The 5-stage loop

| # | Stage | Model | Tools | Retries | Gate |
|---|-------|-------|-------|---------|------|
| 1 | Interview | Opus 4.7 | none | 1 | n/a |
| 2 | Codegen | Sonnet 4.6 → Opus on bust | compile | 3 + 1 escalation | n/a |
| 3 | Correctness | Sonnet 4.6 → Opus on bust | compile, correctness | 3 + 1 escalation | **HARD** |
| 4 | Performance | Sonnet 4.6 → Opus on bust | compile, correctness, profile | 3 + 1 escalation | soft |
| 5 | Polish | Sonnet 4.6 | none | 1 | n/a |

**Hard gate:** Stage 3 failure ⇒ no kernel ships. Returns `SynthesisResult.failed(stage=3, ...)`.
**Soft gate:** Stage 4 below perf bar ⇒ kernel ships with a warning. User decides whether to use it.

### 3.3 Data flow

```
INPUT  → prompt, reference fn, target arch, config
   │
Stage 1  →  KernelSpec  (FROZEN — downstream cannot mutate)
   │
Stage 2  →  KernelArtifact  (kernel.cu, wrapper, compiled .so)
   │
Stage 3  →  CorrectnessReport  (HARD gate: pass or no ship)
   │
Stage 4  →  PerformanceReport  (soft gate: warn but ship)
   │
Stage 5  →  Annotated kernel + final SynthesisReport
   │
OUTPUT → callable kernel + report + run dir with full provenance
```

**Frozen KernelSpec contract:** Stage 1 outputs a structured spec (signature, dtypes, shapes, target arch, latency-vs-throughput, precision tolerance, layout hints, optimization priorities). Downstream stages cannot redefine the problem. If Stage 4 wants to change dtype, it must fail and surface to the user.

**Provenance:** every prompt, every retry, every metric, every error message lands in the run dir. Reproducibility + debuggability + eval-analysis material.

---

## 4. Components

### 4.1 Orchestrator

```python
class Orchestrator:
    def __init__(self, llm: LLMClient, gpu: GPURunner, store: ArtifactStore, cfg: SynthesisConfig): ...

    def run(self, prompt: str, reference: Callable, target: str) -> SynthesisResult:
        run_id = self.store.new_run()
        spec       = Stage1Interview(self.llm).run(prompt, reference, target)
        artifact   = Stage2Codegen(self.llm, self.gpu).run(spec, retry_budget=3)
        correct    = Stage3Correctness(self.llm, self.gpu).run(spec, artifact, reference, retry_budget=3)
        if not correct.passed:
            return SynthesisResult.failed(stage=3, reason=correct)   # HARD GATE
        perf       = Stage4Performance(self.llm, self.gpu).run(spec, artifact, retry_budget=3)
        polished   = Stage5Polish(self.llm).run(spec, artifact, correct, perf)
        return SynthesisResult.ok(polished, correct, perf, run_id=run_id)
```

**Sonnet→Opus escalation:** at the orchestrator level, after a stage exhausts its retry budget on Sonnet, one final attempt is made on Opus. One escalation per stage, max. Logged.

### 4.2 LLMClient — `AnthropicClient`

Single seam to the Anthropic API.

- **Prompt caching.** System prompts (CUDA conventions, target-arch knowledge file, exemplars) marked `cache_control: ephemeral`. User-turn content (the prompt + last error) is uncached. Stage system prompts are 5–15K tokens; cache hit ≈ 10× cheaper, ≈ 2× faster.
- **Tool use, scoped per stage:**
  - `compile_kernel(src) → {ok, errors[], warnings[], ptx_size}`
  - `run_correctness(kernel_so, reference_fn, shapes) → {max_abs_err, max_rel_err, failing_inputs}`
  - `nsight_profile(kernel_so, sample_input) → {regs, occupancy, uncoalesced_pct, spill_bytes, ...}`
  - Stage 1/5: no tools. Stage 2: compile only. Stage 3: compile + correctness. Stage 4: all three.
- **Telemetry.** Every call logs input/output tokens, cache hits, model used, latency, into the run report. Tunes future cost/perf decisions.

### 4.3 GPURunner — `LocalGPURunner`

The boundary between Python and the GPU. **Subprocess-based; never load CUDA into the orchestrator's own process** (one segfault = whole run dies).

- `compile(src) → CompileResult` — `nvcc` in a tempdir, configurable flags (`-O3 -arch=sm_80 --use_fast_math` defaults). Captures stderr. **Caches by `hash(src)`** so identical retries are free.
- `run_kernel(so_path, inputs) → output` — loads `.so` in a child process, runs, returns. Hard 30s timeout; killed on timeout.
- `profile(so_path, inputs) → NsightMetrics` — wraps `ncu --set basic --csv ...`, parses output. **Gracefully degrades** if Nsight isn't installed (Colab often lacks it) — returns partial metrics.

### 4.4 ArtifactStore — `LocalDirStore`

```
~/.cache/cuda_engine/runs/<run_id>/
├── inputs/                  prompt.txt, reference.py, config.json
├── stage1_interview/        prompt_to_llm.md, llm_response.md, kernel_spec.json
├── stage2_codegen/
│   ├── attempt_01/          prompt_to_llm.md, kernel.cu, compile.log, result.json
│   ├── attempt_02/...
│   └── final/               kernel.cu, kernel.so
├── stage3_correctness/...
├── stage4_performance/...
├── stage5_polish/           kernel_annotated.cu
└── report.json              full SynthesisReport
```

Flat directories. JSON + source files. No database in v1. **Privacy note:** prompt traces include full LLM inputs; documented "do not commit run dirs with proprietary references."

### 4.5 Output integration

`torch.utils.cpp_extension.load_inline` — fast setup, no `setup.py`, autograd-aware via `torch.autograd.Function` wrapper for forward-only kernels. Migrate to `torch.library.custom_op` when backward-pass synthesis lands (post-v1).

---

## 5. Error Handling & Retry

### 5.1 Error taxonomy

| Bucket | Examples | Handling |
|--------|----------|----------|
| **Transient external** | Anthropic 429/503, network blip, Nsight subprocess crash | Exp backoff retry (3 attempts: 1s/4s/16s). Not counted against stage budget. |
| **Bounded model failure** | Compile error, numerical mismatch, perf below target | Counted against stage budget. Feed targeted error context back to LLM. |
| **Hard structural** | Reference fails introspection, target arch unsupported, GPU unavailable, OOM at smallest test shape | Fail fast. Clear error naming what's wrong + what user should do. |
| **Bug in our code** | Pydantic validation fail, internal state inconsistent | Raise. Full traceback to run dir. User sees `run_id` + bug-report instruction. |

### 5.2 Retry budgets and cost envelope

| Stage | Retry budget | Escalation | Hard ceiling |
|-------|--------------|------------|--------------|
| 1 Interview | 1 | Sonnet → Opus on first failure | 2 LLM calls |
| 2 Codegen | 3 | Opus on attempt 4 | 4 LLM + 4 compile |
| 3 Correctness | 3 | Opus on attempt 4 | 4 LLM + 4 verify runs |
| 4 Performance | 3 | Opus on attempt 4 | 4 LLM + 4 profile runs |
| 5 Polish | 1 | none | 1 LLM call |

**Worst case:** 12 Sonnet + 4 Opus ≈ $0.50–$2 per kernel (with cache hits).
**Median expected:** ~5 calls, ~$0.10–$0.30.
**All budgets exposed via `SynthesisConfig`.**

### 5.3 Retry context (what gets fed back to the LLM)

Each retry is a fresh conversation turn — **not** an appended chat history. Targeted, cache-friendly, prevents the model re-reading its own bad reasoning.

- **Stage 2 retry:** `KernelSpec` + previous `kernel.cu` + truncated compiler error (first 50 + last 10 lines if huge) + "fix this."
- **Stage 3 retry:** spec + current kernel + correctness diff (max abs/rel err, failing input shapes/values, sample mismatched values).
- **Stage 4 retry:** spec + current kernel + Nsight metrics + actionable hint (e.g., "register pressure 96, target <128"; "occupancy 23%, target >50%"; "uncoalesced loads at lines [N, M]").

### 5.4 User-facing failure UX

- **Library:** `SynthesisResult` with `.passed`, `.failed_stage`, `.report`, `.warnings`. Never raises for bounded failures. Raises only for bugs and config errors.
- **CLI exit codes:** 0 = ship-quality, 1 = correctness fail, 2 = perf-bar miss (kernel still shipped), 3 = hard error.
- **Streamlit demo:** live per-stage status, streaming prompts/responses/errors as the run progresses.

---

## 6. Testing & Eval

### 6.1 Three layers

**Layer 1 — Unit tests (no LLM, no GPU)**
- Stage logic with `MockLLMClient` + `MockGPURunner`.
- Pydantic round-trips. `LocalDirStore` filesystem ops. Prompt builders. Error parsers.
- Runs <30s. **Every PR.** Target ~80% line coverage on stages + orchestrator.

**Layer 2 — Integration tests (real LLM, real GPU, small)**
- ~5 e2e tests on tiny known-easy kernels (vector add, scalar mul, simple sum).
- Real Anthropic + real A100 (Colab CI or self-hosted A100 runner).
- ~$0.10/run, ~5 min/run. **Nightly**, not per-PR. Tagged `@pytest.mark.integration`.

**Layer 3 — Eval suite (the "v1 ships" gate)**

*Internal regression set (~30 kernels):* hand-picked, in-repo, ~20 elementwise/fused + ~10 reductions. Each is a directory: `prompt.txt`, `reference.py`, `expected_shapes.yaml`, `notes.md`. Runs on every release candidate.

*KernelBench subset (~50 kernels):* Stanford KernelBench filtered to v1 scope. Reports speedup distribution; functional pass-rate is the gate.

*Eval runner CLI:*
```bash
cuda-engine eval --suite internal
cuda-engine eval --suite kernelbench
cuda-engine eval --suite both --out evals/2026-04-26/
```
Produces markdown report (per-kernel + aggregate + regression-vs-prev) + CSV. **This is the v1 release artifact.**

### 6.2 Pass criteria for v1 release (the perf bar from Q7)

| Metric | Target |
|--------|--------|
| Functional pass rate (internal) | ≥ 95% |
| Functional pass rate (KernelBench subset) | ≥ 80% |
| Median speedup vs torch.compile | ≥ 1.0× |
| p25 speedup vs torch.compile | ≥ 0.7× |
| % kernels hitting fast_1 (>1× torch.compile) | ≥ 30% |

### 6.3 CI strategy

- **PR CI** — unit + lint + types. <2 min. GitHub Actions, no GPU.
- **Nightly CI** — integration. ~30 min. Colab job or self-hosted A100.
- **Pre-release CI** — full eval. 2–4 hours, ~$5–15 API spend. Manual trigger.

### 6.4 Versioned eval artifacts

Eval kernels live in repo (`evals/internal/`). Eval results versioned at `evals/results/YYYY-MM-DD/`. Reproducibility + easy contribution + no external dataset infra.

---

## 7. Repo Layout

```
cuda-engine/
├── README.md  LICENSE  pyproject.toml  ruff.toml
├── .github/workflows/    {pr.yml, nightly.yml, eval.yml}
├── src/cuda_engine/
│   ├── api.py  cli.py  config.py  orchestrator.py
│   ├── stages/           {base, interview, codegen, correctness, performance, polish}.py
│   ├── services/
│   │   ├── llm/          {base, anthropic, tools}.py
│   │   ├── gpu/          {base, local}.py
│   │   └── store/        {base, local_dir}.py
│   ├── models/           {spec, artifact, reports}.py
│   ├── prompts/          *.md   (real files, PR-reviewed, diffable)
│   └── targets/          sm_80.py / sm_90.py / sm_100.py
├── tests/{unit, integration}/
├── evals/
│   ├── internal/         ~30 kernel dirs
│   ├── kernelbench/      filter scripts + README
│   ├── runner.py
│   └── results/          versioned by date
├── examples/{notebook.ipynb, web_demo.py, kernels/}
└── docs/
    ├── brainstorming-notes.md
    ├── superpowers/{specs/, plans/}
    └── architecture.md
```

**Two non-obvious choices, restated:**
1. `targets/sm_90.py` and `sm_100.py` exist from day 1 (mostly empty) so adding Blackwell verification is "fill in the file," not "rewrite codegen."
2. `prompts/*.md` are real markdown files, PR-reviewable. Prompt engineering is engineering.

**Stack pinned for v1:**
- Python 3.11+
- `anthropic` (latest, prompt-cache support)
- `torch` 2.4+ (cpp_extension stable)
- `pydantic` v2
- `typer` (CLI)
- `pytest` + `pytest-asyncio`
- `ruff` + `mypy`
- Streamlit only in `examples/`, never in core.

---

## 8. Milestones & Checkpoints

The path from empty repo to v1 release. Each milestone has explicit **definition of done**. Do not move past a checkpoint until it is met. These are architecture-level milestones; the implementation plan (separate doc, written next via `writing-plans`) breaks them into 2–5 minute steps.

### Milestone 0 — Skeleton (Week 1)

**Build**
- Repo created, `pyproject.toml`, ruff/mypy/pytest configured.
- All ABCs and Pydantic data classes defined: `LLMClient`, `GPURunner`, `ArtifactStore`, `KernelSpec`, `KernelArtifact`, reports, `SynthesisConfig`, `SynthesisResult`.
- Mock implementations: `MockLLMClient`, `MockGPURunner`, in-memory `MockStore`.
- Empty `Orchestrator` and 5 empty `Stage` classes that pass through.
- PR CI green (`pytest tests/unit`, ruff, mypy).

**Checkpoint M0 — must pass before M1:**
- [ ] `pip install -e .` works.
- [ ] `pytest tests/unit` is green with ≥1 test per module.
- [ ] `ruff check .` and `mypy src/` are clean.
- [ ] A dummy `synthesize("noop", lambda x: x, "sm_80")` returns a `SynthesisResult.ok` from end to end using mocks (no Anthropic, no GPU).
- [ ] PR CI workflow runs on push.

### Milestone 1 — Stage 2 alone (codegen + compile, real Anthropic, real A100) (Weeks 1–2)

**Build**
- `AnthropicClient` with prompt caching + the `compile_kernel` tool.
- `LocalGPURunner.compile()` using subprocess `nvcc` + hash-based cache.
- `Stage2Codegen` real implementation, including 3 retries with compiler-error feedback.
- One trivial kernel works end-to-end: vector add `lambda x, y: x + y` on `sm_80`.
- Run dir layout under `~/.cache/cuda_engine/runs/<run_id>/` populated.

**Checkpoint M1 — must pass before M2:**
- [ ] On Colab A100: `synthesize("element-wise add two fp32 vectors", vector_add_ref, "sm_80")` produces a compiled `kernel.so` and the orchestrator returns `SynthesisResult` with stage 2 marked complete (stages 3–5 still stubbed).
- [ ] Compile cache works: identical retry is a cache hit (verified by log).
- [ ] Prompt cache works: 2nd run on a similar prompt shows cache_read_input_tokens > 0 in run report.
- [ ] Run dir contains `stage2_codegen/attempt_01/...` with prompt + kernel.cu + compile.log.
- [ ] Compile-error retry path verified: induce a deliberately bad prompt; observe attempt_02 fixes it.
- [ ] Integration test for vector-add passes.

### Milestone 2 — Full pipeline correctness (Weeks 2–3)

**Build**
- `Stage1Interview` (Opus 4.7, no tools).
- `Stage3Correctness` real impl with `run_correctness` tool, 3 retries with diff feedback.
- `Stage5Polish` real impl (annotation + report assembly).
- `Stage4Performance` stubbed: just runs benchmark vs torch.compile, no Nsight loop.
- 5 simplest kernels from internal regression set passing functional: vector add, scalar multiply, fp16 RMSNorm, sum reduction, argmax.
- Hard-gate semantics enforced: induced numerical fault returns `SynthesisResult.failed(stage=3)`.

**Checkpoint M2 — must pass before M3:**
- [ ] 5/5 simplest internal kernels pass functional on Colab A100.
- [ ] Stage 3 hard gate verified: deliberately broken kernel does NOT ship; user sees `failed(stage=3)` with diff in report.
- [ ] Stage 1 produces a `KernelSpec` and Stage 2 honors it (assertion: stage 2 doesn't override dtype/shape).
- [ ] `Stage5Polish` produces a Doxygen-annotated kernel + a complete `SynthesisReport` (markdown + json).
- [ ] Run cost telemetry works: report shows tokens, cache hits, latency per stage.

### Milestone 3 — Performance loop (Weeks 3–4)

**Build**
- `nsight_profile` tool wired up; graceful degradation when Nsight absent.
- `Stage4Performance` real impl: benchmark → if < target, profile → feed metrics back to LLM, retry up to 3.
- nvcc flags tuned per target (`-O3 -arch=sm_80 --use_fast_math` baseline; future arch flags configurable).
- `cuda-engine eval --suite internal` CLI command working on the 30-kernel internal set.
- Sonnet→Opus escalation wired in orchestrator.

**Checkpoint M3 — must pass before M4:**
- [ ] All 30 internal kernels run end-to-end.
- [ ] ≥ 25/30 (≥83%) functional pass rate on internal set.
- [ ] ≥ 10/30 (≥33%) hit fast_1 (>1× torch.compile).
- [ ] At least one kernel demonstrably improved by Nsight feedback (attempt 2 perf > attempt 1; logged in report).
- [ ] At least one kernel demonstrably escalated to Opus and succeeded (logged in report).
- [ ] Eval CLI produces markdown + CSV report at `evals/results/<date>/`.

### Milestone 4 — v1 release readiness (Weeks 4–6)

**Build**
- All retry budgets tuned per real failure-mode data from M3.
- `cli.py` complete: `synthesize`, `eval`, `inspect` commands.
- `examples/web_demo.py` Streamlit app working.
- `examples/notebook.ipynb` runs cleanly on a fresh Colab.
- README with quickstart, demo GIF, eval numbers.
- Nightly + pre-release CI workflows green.
- KernelBench subset filtered + scripted.
- LICENSE chosen, security/privacy docs written (prompt-trace warning).

**Checkpoint M4 — v1 ships:**
- [ ] **Internal regression set:** ≥ 95% functional pass, median speedup ≥ 1.0× torch.compile, p25 ≥ 0.7×, ≥ 30% fast_1.
- [ ] **KernelBench subset:** ≥ 80% functional pass rate.
- [ ] All three CI workflows are green.
- [ ] `pip install cuda-engine` from TestPyPI works on a fresh Colab.
- [ ] Streamlit demo runs end-to-end on a fresh machine.
- [ ] README has runnable quickstart, eval numbers, and a one-paragraph honest scope statement.
- [ ] At least 3 worked examples in `examples/kernels/` with full provenance.
- [ ] Privacy + cost docs written.
- [ ] Eval results for v1.0 committed at `evals/results/v1.0-<date>/`.

### Post-v1 roadmap (not gated, just signposted)

- **v1.1** — Pluggable LLM backend (LiteLLM adapter), more eval kernels.
- **v1.5** — Run resumability, persistent run DB, autotuning sweeps, Hopper (`sm_90`) verification on rented hours.
- **v2.0** — Blackwell (`sm_100`) verification, GEMM/matmul tile category, remote `GPURunner`.
- **v3.0** — Migration mode (port kernel from arch X to arch Y), attention category, multi-GPU.
- **Research (no commitment):** formal verification, multi-arch cost modeling, rack-scale.

---

## 9. Open Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Colab A100 unavailability blocks dev/CI | High | Self-hosted A100 fallback (Lambda/Vast hourly ~$1.20/hr). Document Colab caveats. |
| Anthropic API cost during eval runs | Medium | Pre-release eval is manual-trigger only. Tight retry budgets. Aggressive prompt caching. |
| Stage 4 perf loop converges slowly / doesn't converge | Medium | Bounded retries. Soft gate (kernel ships even if slow). Nsight feedback may need iteration on prompt design. |
| Numerical tolerance choices mask real bugs | Medium | Tolerance is per-spec (Stage 1 sets it); default tight (rtol=1e-3, atol=1e-3 for fp16). Failing-input samples in report. |
| Reference-function introspection brittle (closures, decorators) | Low | Document "reference must be a plain function with type hints." Test with diverse decorators in unit tests. |
| Streamlit demo dependency surface bloats v1 | Low | Demo is in `examples/`, optional dep group. Core never imports Streamlit. |
| KernelBench license / distribution constraints | Low | Subset is filter-scripted, not vendored. Document fetch step in `evals/kernelbench/README.md`. |
| Prompt-trace artifacts leak proprietary references | Low | Documented warning in README + run-dir README. `cuda-engine inspect` could redact in v1.5. |

---

## 10. Glossary

- **fast_1** — KernelBench convention for "speedup > 1× over `torch.compile` baseline."
- **fast_2** — speedup > 2× over `torch.compile`.
- **Hard gate** — failure halts the pipeline; no kernel ships.
- **Soft gate** — failure continues the pipeline with a warning attached to the result.
- **KernelSpec** — frozen structured contract output by Stage 1; downstream stages cannot mutate.
- **Provenance** — the full trail of prompts, responses, retries, errors, and metrics preserved per run.

---

## 11. Approval

- [x] Section 1 — Component Map (approved 2026-04-26)
- [x] Section 2 — Data Flow (approved 2026-04-26)
- [x] Section 3 — Components Detail (approved 2026-04-26)
- [x] Section 4 — Error Handling & Retry (approved 2026-04-26)
- [x] Section 5 — Testing & Eval (approved 2026-04-26)
- [x] Section 6 — Repo Layout (approved 2026-04-26)
- [x] Section 8 — Milestones & Checkpoints (approved 2026-04-26)

Design frozen. Implementation plan to be written next via the `writing-plans` skill at `docs/superpowers/plans/2026-04-26-cuda-synthesis-engine-plan.md`.
