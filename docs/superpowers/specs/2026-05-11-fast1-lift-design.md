# Stage 4 fast_1 Lift — Design Document

| Field | Value |
|---|---|
| **Status** | Approved (2026-05-11) |
| **Owner** | Shivnarain |
| **Branch** | `m3/perf-loop` |
| **Parent design** | `docs/superpowers/specs/2026-04-26-cuda-synthesis-engine-design.md` |
| **Implementation plan** | `docs/superpowers/plans/2026-05-11-fast1-lift-plan.md` (to be created) |
| **Milestone** | M3 — fast_1 ≥ 10/30 checkpoint |

---

## 1. Goal

Lift `fast_1` (kernels with median speedup strictly `>1.0×` vs `torch.compile`) from the current `1–4/30` to the M3 gate of `≥10/30`. Targeted scope: the **plateau cluster** — 6+ kernels (`layernorm_fp16`, `rms_norm_fp16`, `rmsnorm_silu_fused_fp16`, `softmax_lastdim_fp16`, `softmax_numerator_fp16`, `masked_mean_fp16`, `geglu_fp16`, and possibly more) that land at exactly `1.00×` because Stage 4's retry loop exits the moment target is met.

Out of scope: the stuck-below cluster (`sigmoid_mul_fp16` 0.77, `tanh_add_fp32` 0.81, `bias_gelu_fp16` 0.99). Those represent torch.compile-is-genuinely-hard-to-beat territory and need a separate, deeper engineering pass if M3 doesn't close after this lift.

---

## 2. Scope

### In scope
- Remove the `if current_speedup >= target: break` early-exit in `Stage4Performance._retry_loop`.
- Add best-so-far tracking inside the loop. Continue feeding the LATEST attempt to the LLM for iterative refinement, but RETURN the BEST attempt at function end.
- Append ~12 lines to `src/cuda_engine/prompts/perf_fix.md` giving the LLM concrete "beat, don't match" levers (vectorization, wave alignment, ILP, fusion, register pressure).
- Unit tests covering best-so-far semantics: regression protection, budget-exhaustion past target, tie behavior.
- One prompt-content assertion test.

### Out of scope
- Dynamic prompt branching (was Approach C in brainstorm). Deferred until eval data shows B is insufficient.
- Two-tier soft/stretch target config knob. Same reason.
- Anything addressing the stuck-below cluster.
- Bumping `opus_retry_budget_performance` above 1. Plateau kernels already meet target on Sonnet — Opus doesn't help them.

### Explicitly rejected
- **Raising `perf_target_speedup_vs_torch_compile` from 1.0 to 1.10.** Conflates "didn't beat parity" with "didn't reach stretch goal" in the `below_target` semantics. The retry loop and the soft-gate both read the same config field today; bumping it forces both to move together.

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage4Performance._retry_loop                                      │
│                                                                     │
│  Before:                          After:                            │
│  ──────                           ─────                             │
│  for attempt in range(N):         best_artifact = artifact          │
│      if speedup >= target:        best_benchmark = benchmark        │
│          break  ← exits here      best_speedup = speedup            │
│      ...                          for attempt in range(N):          │
│      current = new                    ...  # always iterate         │
│  return current                       current = new  # feed latest  │
│                                       if new_speedup > best_speedup:│
│                                           best = (current snapshot) │
│                                   return best                       │
└─────────────────────────────────────────────────────────────────────┘
```

**Two complementary changes:**
1. **Loop fix** — surgical change to use already-budgeted retries that today are wasted on parity-hitters.
2. **Prompt nudge** — gives the LLM a concrete new target to chase on iterations 2 and 3.

Either alone is weaker. The loop fix without the prompt change means the LLM produces similar kernels on each iteration and we rely on stochastic variance. The prompt change without the loop fix means the LLM gets the "beat it" advice but the loop exits before it can act on the advice for plateau kernels.

---

## 4. Components

### 4.1 `Stage4Performance._retry_loop` modifications

**File:** `src/cuda_engine/stages/performance.py`

Signature unchanged. Body changes:

```python
def _retry_loop(
    self, *, spec, artifact, benchmark, speedup, target, inputs, run_id,
    retry_budget, model, attempt_offset=0,
):
    # ... existing setup unchanged ...

    # NEW: best-so-far tracking, initialized from input state
    best_artifact = current_artifact
    best_benchmark = current_benchmark
    best_speedup = current_speedup

    for local_attempt in range(1, retry_budget + 1):
        attempt = local_attempt + attempt_offset
        # REMOVED: if current_speedup >= target: break

        # ... existing attempt body unchanged through new_speedup computation ...

        # MODIFIED: note now records best alongside current
        notes.append(
            f"perf_repair attempt {attempt}: speedup {current_speedup:.3f} -> "
            f"{new_speedup:.3f} (best={max(best_speedup, new_speedup):.3f})"
        )

        # Always update current (feeds the LLM continuity for next iteration)
        current_artifact = candidate
        current_benchmark = new_benchmark
        current_speedup = new_speedup

        # NEW: only update best on strict improvement
        if new_speedup > best_speedup:
            best_artifact = candidate
            best_benchmark = new_benchmark
            best_speedup = new_speedup

    # NEW: terminal warning now compares best, not current
    if best_speedup < target:
        warnings.append(
            f"perf retry budget exhausted: best speedup {best_speedup:.3f} "
            f"below target {target:.3f}"
        )

    return best_artifact, best_benchmark, best_speedup, warnings, notes
```

### 4.2 `perf_fix.md` prompt addition

**File:** `src/cuda_engine/prompts/perf_fix.md`

Insert after the existing "Optimization themes to consider" bullet block, before the "Output the complete revised CUDA source..." closing instruction:

```markdown
Matching torch.compile is acceptable but not the goal. To strictly beat it on A100:
- Prefer vectorized memory ops: `float4` for fp32, `__half2` for fp16. They double effective bandwidth on aligned contiguous data.
- Align grid to A100's 108 SMs. A full wave is a multiple of 108 blocks; a partial tail wave wastes runtime. For tensors that don't divide evenly, prefer fewer-larger blocks over more-smaller.
- Maximize instruction-level parallelism: `#pragma unroll` inner loops with small bounded trip count. Keep enough independent work per thread to hide arithmetic and memory latency.
- Fuse passes when the KernelSpec permits. Reductions followed by elementwise can often be one-pass with `__shfl_down_sync` warp reductions.
- Inspect register pressure first if Nsight shows occupancy < 50%. If regs/thread > 64 on a 256-thread block, work-split or block-size reduction frees waves.
```

---

## 5. Data flow

**Per-attempt flow inside `_retry_loop`** (new):
```
1. LLM gets current state (latest attempt's source + bench + metrics + hints)
2. LLM returns new CUDA source
3. Compile + benchmark
4. Compute new_speedup
5. Note: "attempt N: current → new (best=max)"
6. Always: current = new (feed continuity)
7. Conditional: if new_speedup > best_speedup: best = new (protect winner)
8. Continue to next iteration regardless of target
9. After budget: return best (not current)
```

**Stage 4 → orchestrator** (unchanged):
- `Stage4Performance.run()` calls `_retry_loop` exactly as before.
- The returned tuple has the same shape; only the values differ (best instead of last).
- `below_target` is set from the returned `current_speedup` (which is now `best_speedup` semantically).
- Existing escalation block (Sonnet → Opus) uses returned best as input to the Opus loop. Opus iterations build on Sonnet's best, not Sonnet's last.

---

## 6. Trace and observability

- Per-attempt `notes` now show `(best=X.XXX)` alongside the transition, making it visible in `report.json` which iteration won.
- Terminal warning text changes from `final speedup` to `best speedup`. Cosmetic but clearer.
- No schema change to `PerformanceReport`. `speedup_vs_torch_compile` continues to be the final speedup (= best from the loop).

---

## 7. Error handling

| Scenario | Behavior |
|---|---|
| Sonnet retry attempt produces a worse kernel | `current` becomes the worse one (for LLM continuity), `best` is preserved, returned state = best |
| All retries are worse than initial | `best` stays at initial values, returned state = initial. Exactly the same outcome as before this change. |
| All retries fail to compile | `best` stays at initial, no `best_*` updates fire. Same as today. |
| First retry > target by a lot, then bounces lower | Returns the first-retry winner. New behavior — previously would have early-exited at first retry, returning the same kernel. |
| Tied speedup between two attempts | First-wins (strict `>`). Avoids churn replacing equivalent kernels. |

---

## 8. Testing

### Unit
**`tests/unit/stages/test_performance.py` — 3 new tests:**

- `test_perf_retry_loop_returns_best_so_far_on_regression`
  - Setup: retry_budget=3, mock benchmarks producing speedups 1.0 → 1.10 → 0.90 (initial baseline lower so we enter the loop).
  - Assert: returned `speedup == 1.10`, returned artifact corresponds to attempt 2's recompile, attempt 3's recompile is persisted but not returned.

- `test_perf_retry_loop_exhausts_budget_past_target`
  - Setup: retry_budget=3, target=1.0, mock benchmarks 1.0 → 1.0 → 1.05.
  - Assert: LLM `complete()` called 3 times (no early exit), returned speedup == 1.05, attempt dirs `01/02/03` all exist.

- `test_perf_retry_loop_first_wins_on_tie`
  - Setup: retry_budget=3, mocks producing 1.05 → 1.05 → 1.00.
  - Assert: returned speedup == 1.05, returned artifact corresponds to attempt 1 (tie does not replace).

**`tests/unit/test_prompts.py` — extend existing:**

- `test_perf_fix_prompt_includes_beat_torch_compile_guidance`
  - Loads `perf_fix` prompt and asserts the new key phrases appear:
    - `"Matching torch.compile is acceptable but not the goal"`
    - `"float4"` and `"__half2"`
    - `"108 SMs"`
    - `"#pragma unroll"`

### Integration
- No new integration tests. Existing `test_e2e_perf_loop_escalation.py` exercises Stage 4 end-to-end; loop behavior change is fully covered by unit tests.

### Eval validation (the real measurement)
- After commit, rerun `cuda-engine eval --suite internal --resume` on Colab A100.
- Success criteria:
  - **Minimum (proves the change moves the needle):** `fast_1 ≥ 5/30`.
  - **Stretch (closes M3 gate):** `fast_1 ≥ 10/30`.
  - **Regression guard:** `pass_rate ≥ 28/30` (no kernel that previously passed should now fail).
- If minimum is met but stretch isn't, escalate to Approach C (dynamic prompt branching) in a follow-up brainstorm.

---

## 9. Commit plan (preview)

Detailed steps in the implementation plan. High-level:

1. `feat(stage4): best-so-far tracking + exhaust budget past parity` — loop change + 3 unit tests.
2. `feat(prompt): beat-torch-compile levers in perf_fix` — prompt edit + prompt-content test.
3. `docs(m3): post-fast1-lift eval evidence` — after Colab rerun, append to `M3-evidence.md`.

Aim: 3 commits, ~120 LOC + tests + prompt + evidence.

---

## 10. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| LLM bounces around 1.0× across 3 attempts without crossing | Medium | best-so-far protects against regression; budget bounded so cost capped |
| Sonnet→Opus escalation summary describes Sonnet's LAST while artifact is Sonnet's BEST | Low | Acceptable: Opus benefits from seeing what worked + summary tells it what to fix |
| Prompt nudge biases LLM toward parallel ops on reduction-heavy workloads | Low | Themes framed as "levers" not commandments; existing Nsight hints still primary signal |
| Cost increase from 3 attempts always running vs ~1 attempt for parity-hitters | Medium | This is the explicit trade — extra Stage 4 LLM cost in exchange for fast_1 lift. Acceptable for v1. |

---

## 11. Open questions

None at design time. Surfaces during execution get logged in the plan.
