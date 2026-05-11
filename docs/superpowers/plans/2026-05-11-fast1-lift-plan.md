# Stage 4 fast_1 Lift Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push Stage 4's perf retry loop past `1.0×` parity by removing the early-exit when target is met, tracking best-so-far across attempts, and giving the LLM a concrete "beat torch.compile" prompt. Targets the plateau cluster (6+ kernels at exactly 1.00×) to lift fast_1 from 1/30 toward the M3 ≥10/30 gate.

**Architecture:** Two coordinated changes. (1) `Stage4Performance._retry_loop` removes its `current_speedup >= target` early-exit and adds best-so-far tracking — the loop now exhausts its budget and returns the best attempt, while feeding the LATEST attempt to the LLM for iterative refinement. (2) `perf_fix.md` gains a "Matching torch.compile is acceptable but not the goal" paragraph with A100-specific levers (vectorization, wave alignment, ILP, fusion, register pressure).

**Tech Stack:** Python 3.11+, pytest + `MockLLMClient` / `MockGPURunner` for unit tests. No new deps.

**Spec:** `docs/superpowers/specs/2026-05-11-fast1-lift-design.md`

**Skills referenced:**
- @superpowers:test-driven-development — every code task: failing test first, then minimal code, then commit.
- @superpowers:verification-before-completion — checkpoints pass only with real command output.
- @superpowers:systematic-debugging — when a step fails, diagnose root cause; don't paper over.

**Branch:** `m3/perf-loop` (HEAD `319c468` after spec commit).

---

## File Structure (locked)

### Files modified
| Path | Change |
|---|---|
| `src/cuda_engine/stages/performance.py` | `_retry_loop`: remove early-exit, add best-so-far tracking, return best instead of current. Terminal warning compares best. Notes record `(best=X.XXX)` per attempt. |
| `src/cuda_engine/prompts/perf_fix.md` | Append "Matching torch.compile is acceptable but not the goal" block (~12 lines) before the closing "Output the complete..." instruction. |
| `tests/unit/stages/test_performance.py` | Update 2 existing tests (retry_budget adjustments) + add 3 new tests for best-so-far semantics. |
| `tests/unit/test_prompts.py` | Add 1 assertion test for new perf_fix content. |

### Files created
None.

---

## Pre-flight: existing test updates required

Before implementing the loop change, two existing tests in `tests/unit/stages/test_performance.py` need their `retry_budget` adjusted to keep their semantics under the new "no early-exit" behavior. Both currently rely on the early-exit firing when target is met mid-loop; without it, they'd try to run more attempts than the mock has responses queued for and fail with `MockLLMClient: no canned responses left`.

These are not breaking changes to the test intent — just adjustments so the budget naturally aligns with the queued mock data.

| Test | Current `retry_budget` | New `retry_budget` | Why |
|---|---|---|---|
| `test_stage4_performance_retries_until_target_met` | 3 | 1 | Test queues 1 LLM response + 1 compile + 2 benchmarks (initial + after-attempt-1). Today loop early-exits at attempt 1; new behavior would try attempt 2. Setting budget to 1 keeps the original intent (one retry succeeds) and matches the queued mock data. |
| `test_stage4_performance_records_failed_compile_in_warnings_and_continues` | 3 | 2 | Test queues 2 LLM responses + 2 compile (1st fails, 2nd succeeds) + 2 benchmarks. Today early-exits at attempt 2 (hits 2.0×). New behavior would try attempt 3. Budget 2 keeps intent (failed compile → next attempt succeeds). |

Both edits are one-line changes in the `.run(... retry_budget=N)` call. Do them in Chunk 1 alongside the loop change.

---

## Chunk 1: Loop change — remove early-exit, add best-so-far tracking

Single self-contained code change to `Stage4Performance._retry_loop`. TDD-driven: write the three new behavior tests first, watch them fail, implement, watch them pass.

### Task 1.1: Add `test_perf_retry_loop_exhausts_budget_past_target` (RED)

**Files:**
- Modify: `tests/unit/stages/test_performance.py` (append new test)

- [ ] **Step 1: Read existing test helpers**

Skim `tests/unit/stages/test_performance.py` lines 1–170 to identify the helpers in use:
- `_initial_artifact_in_store(store, run_id, src)` — creates initial KernelArtifact and writes it to `InMemoryStore`.
- `_llm_compile_response(src)` — returns an `LLMResponse` with a compile_kernel tool call.
- `_spec()` — returns a `KernelSpec` for the standard test scenario.

Confirm `MockGPURunner` takes `benchmark_results`, `profile_results`, `compile_results` queues.

- [ ] **Step 2: Append failing test**

Append to `tests/unit/stages/test_performance.py`:

```python
def test_perf_retry_loop_exhausts_budget_past_target() -> None:
    """When attempt 1 already meets target, loop continues to attempt 3 to push past parity."""
    store = InMemoryStore()
    artifact = _initial_artifact_in_store(store, "run123", "// initial")
    initial = BenchmarkResult(
        ok=True, custom_ms=4.0, baseline_ms=2.0, warmup_iterations=10, timed_iterations=50
    )
    at_target = BenchmarkResult(
        ok=True, custom_ms=2.0, baseline_ms=2.0, warmup_iterations=10, timed_iterations=50
    )
    above_target = BenchmarkResult(
        ok=True, custom_ms=1.9, baseline_ms=2.0, warmup_iterations=10, timed_iterations=50
    )
    gpu = MockGPURunner(
        compile_results=[
            CompileResult(ok=True, so_path=Path("/tmp/v1.so"), log="ok"),
            CompileResult(ok=True, so_path=Path("/tmp/v2.so"), log="ok"),
            CompileResult(ok=True, so_path=Path("/tmp/v3.so"), log="ok"),
        ],
        benchmark_results=[initial, at_target, at_target, above_target],
        profile_results=[NsightMetrics(occupancy=0.5, regs_per_thread=64)] * 3,
    )
    llm = MockLLMClient(
        [
            _llm_compile_response("// v1 at parity"),
            _llm_compile_response("// v2 at parity"),
            _llm_compile_response("// v3 beats parity"),
        ]
    )
    cfg = SynthesisConfig(perf_target_speedup_vs_torch_compile=1.0)
    stage = Stage4Performance(llm=llm, gpu=gpu, store=store, cfg=cfg)

    report, _ = stage.run(spec=_spec(), artifact=artifact, run_id="run123", retry_budget=3)

    assert llm.call_count == 3, "loop should not early-exit when target met mid-loop"
    assert report.speedup_vs_torch_compile > 1.0
    assert abs(report.speedup_vs_torch_compile - (2.0 / 1.9)) < 1e-6
```

- [ ] **Step 3: Run to confirm RED**

```
python -m pytest tests/unit/stages/test_performance.py::test_perf_retry_loop_exhausts_budget_past_target -v
```

Expected: FAIL because the loop early-exits at attempt 1 (when speedup becomes 1.0 == target). `llm.call_count` will be 1, not 3.

### Task 1.2: Add `test_perf_retry_loop_returns_best_so_far_on_regression` (RED)

**Files:**
- Modify: `tests/unit/stages/test_performance.py` (append)

- [ ] **Step 1: Append failing test**

Append to `tests/unit/stages/test_performance.py`:

```python
def test_perf_retry_loop_returns_best_so_far_on_regression() -> None:
    """Attempt 2 wins (1.10x), attempt 3 regresses (0.90x). Return attempt 2."""
    store = InMemoryStore()
    artifact = _initial_artifact_in_store(store, "run123", "// initial")
    initial = BenchmarkResult(
        ok=True, custom_ms=4.0, baseline_ms=2.0, warmup_iterations=10, timed_iterations=50
    )  # 0.5x triggers retry
    at_target = BenchmarkResult(
        ok=True, custom_ms=2.0, baseline_ms=2.0, warmup_iterations=10, timed_iterations=50
    )  # 1.0x
    best = BenchmarkResult(
        ok=True, custom_ms=2.0 / 1.10, baseline_ms=2.0, warmup_iterations=10, timed_iterations=50
    )  # 1.10x
    regress = BenchmarkResult(
        ok=True, custom_ms=2.0 / 0.90, baseline_ms=2.0, warmup_iterations=10, timed_iterations=50
    )  # 0.90x
    gpu = MockGPURunner(
        compile_results=[
            CompileResult(ok=True, so_path=Path("/tmp/v1.so"), log="ok"),
            CompileResult(ok=True, so_path=Path("/tmp/v2.so"), log="ok"),
            CompileResult(ok=True, so_path=Path("/tmp/v3.so"), log="ok"),
        ],
        benchmark_results=[initial, at_target, best, regress],
        profile_results=[NsightMetrics(occupancy=0.5, regs_per_thread=64)] * 3,
    )
    llm = MockLLMClient(
        [
            _llm_compile_response("// v1 at parity"),
            _llm_compile_response("// v2 best"),
            _llm_compile_response("// v3 regression"),
        ]
    )
    cfg = SynthesisConfig(perf_target_speedup_vs_torch_compile=1.0)
    stage = Stage4Performance(llm=llm, gpu=gpu, store=store, cfg=cfg)

    report, final_artifact = stage.run(
        spec=_spec(), artifact=artifact, run_id="run123", retry_budget=3
    )

    assert llm.call_count == 3
    assert abs(report.speedup_vs_torch_compile - 1.10) < 1e-3, (
        f"expected best (1.10x) to be returned, got {report.speedup_vs_torch_compile}"
    )
    # Returned artifact should be the v2 (best) kernel
    assert store._files[("run123", "stage4_performance/perf_repair/attempt_02/kernel.cu")] == b"// v2 best"
    # v3's attempt dir is also written for inspection
    assert ("run123", "stage4_performance/perf_repair/attempt_03/kernel.cu") in store._files
```

- [ ] **Step 2: Run to confirm RED**

```
python -m pytest tests/unit/stages/test_performance.py::test_perf_retry_loop_returns_best_so_far_on_regression -v
```

Expected: FAIL. Today the loop blindly replaces `current_speedup` with each attempt's result, so the final returned speedup would be 0.90 (v3), not 1.10 (v2).

### Task 1.3: Add `test_perf_retry_loop_first_wins_on_tie` (RED)

**Files:**
- Modify: `tests/unit/stages/test_performance.py` (append)

- [ ] **Step 1: Append failing test**

```python
def test_perf_retry_loop_first_wins_on_tie() -> None:
    """Attempt 1 wins (1.05x), attempt 2 ties (1.05x), attempt 3 worse (1.00x). Return attempt 1."""
    store = InMemoryStore()
    artifact = _initial_artifact_in_store(store, "run123", "// initial")
    initial = BenchmarkResult(
        ok=True, custom_ms=4.0, baseline_ms=2.0, warmup_iterations=10, timed_iterations=50
    )
    fast = BenchmarkResult(
        ok=True, custom_ms=2.0 / 1.05, baseline_ms=2.0, warmup_iterations=10, timed_iterations=50
    )
    tie = BenchmarkResult(
        ok=True, custom_ms=2.0 / 1.05, baseline_ms=2.0, warmup_iterations=10, timed_iterations=50
    )
    parity = BenchmarkResult(
        ok=True, custom_ms=2.0, baseline_ms=2.0, warmup_iterations=10, timed_iterations=50
    )
    gpu = MockGPURunner(
        compile_results=[
            CompileResult(ok=True, so_path=Path("/tmp/v1.so"), log="ok"),
            CompileResult(ok=True, so_path=Path("/tmp/v2.so"), log="ok"),
            CompileResult(ok=True, so_path=Path("/tmp/v3.so"), log="ok"),
        ],
        benchmark_results=[initial, fast, tie, parity],
        profile_results=[NsightMetrics(occupancy=0.5, regs_per_thread=64)] * 3,
    )
    llm = MockLLMClient(
        [
            _llm_compile_response("// v1 fast"),
            _llm_compile_response("// v2 tie"),
            _llm_compile_response("// v3 parity"),
        ]
    )
    cfg = SynthesisConfig(perf_target_speedup_vs_torch_compile=1.0)
    stage = Stage4Performance(llm=llm, gpu=gpu, store=store, cfg=cfg)

    report, final_artifact = stage.run(
        spec=_spec(), artifact=artifact, run_id="run123", retry_budget=3
    )

    assert abs(report.speedup_vs_torch_compile - 1.05) < 1e-3
    # First-wins: attempt_01 should be the returned kernel
    assert store._files[("run123", "stage4_performance/perf_repair/attempt_01/kernel.cu")] == b"// v1 fast"
```

- [ ] **Step 2: Run to confirm RED**

```
python -m pytest tests/unit/stages/test_performance.py::test_perf_retry_loop_first_wins_on_tie -v
```

Expected: FAIL — loop early-exits at attempt 1 today (hits target), but the assertion that follows depends on the loop's full behavior; even with early-exit removed and the new behavior, "first wins on tie" requires strict `>` comparison which we haven't implemented yet.

### Task 1.4: Update existing tests' `retry_budget` to align with new loop semantics

**Files:**
- Modify: `tests/unit/stages/test_performance.py:194` and `tests/unit/stages/test_performance.py:265`

- [ ] **Step 1: Update `test_stage4_performance_retries_until_target_met`**

In the body of `test_stage4_performance_retries_until_target_met`, change:

```python
report, final_artifact = stage.run(
    spec=_spec(), artifact=artifact, run_id="run123", retry_budget=3
)
```

to:

```python
report, final_artifact = stage.run(
    spec=_spec(), artifact=artifact, run_id="run123", retry_budget=1
)
```

The `assert llm.call_count == 1` already in this test stays — now it's enforced by budget rather than early-exit.

- [ ] **Step 2: Update `test_stage4_performance_records_failed_compile_in_warnings_and_continues`**

Change:

```python
report, _ = stage.run(spec=_spec(), artifact=artifact, run_id="run123", retry_budget=3)
```

to:

```python
report, _ = stage.run(spec=_spec(), artifact=artifact, run_id="run123", retry_budget=2)
```

Test intent preserved: attempt 1 fails compile, attempt 2 succeeds at 2.0×, loop ends from budget exhaustion.

- [ ] **Step 3: Run both updated tests to confirm they still pass (and DO pass under current code)**

```
python -m pytest tests/unit/stages/test_performance.py::test_stage4_performance_retries_until_target_met tests/unit/stages/test_performance.py::test_stage4_performance_records_failed_compile_in_warnings_and_continues -v
```

Expected: PASS for both. The budget adjustment is invisible to current code (early-exit hides it).

### Task 1.5: Implement loop change (remove early-exit + best-so-far tracking)

**Files:**
- Modify: `src/cuda_engine/stages/performance.py:137-262` (the `_retry_loop` method)

- [ ] **Step 1: Apply the loop change**

In `src/cuda_engine/stages/performance.py`, inside `_retry_loop`:

**Add best-so-far initialization right after the `current_*` initialization** (after current line 159):

```python
        best_artifact = current_artifact
        best_benchmark = current_benchmark
        best_speedup = current_speedup
```

**Remove the early-exit block** (current lines 170–171):

```python
            if current_speedup >= target:
                break
```

Delete those two lines entirely. Keep all other gates in the loop body (the `current_artifact.kernel_so_path is None` check, the source-unreadable check, etc.).

**Update the `notes.append(...)` line** (currently at the end of the success path, around line 263):

Current line:
```python
notes.append(
    f"perf_repair attempt {attempt}: speedup {current_speedup:.3f} -> {new_speedup:.3f}"
)
```

New line (must be computed BEFORE `current_speedup` is overwritten):
```python
notes.append(
    f"perf_repair attempt {attempt}: speedup {current_speedup:.3f} -> "
    f"{new_speedup:.3f} (best={max(best_speedup, new_speedup):.3f})"
)
```

**Keep the unconditional `current_*` update** (already in code):

```python
current_artifact = candidate
current_benchmark = new_benchmark
current_speedup = new_speedup
```

**Immediately after, add the best-so-far update with STRICT `>` comparison** (so tie doesn't replace):

```python
if new_speedup > best_speedup:
    best_artifact = candidate
    best_benchmark = new_benchmark
    best_speedup = new_speedup
```

**Update the terminal warning** (currently around line 273):

Current:
```python
if current_speedup < target:
    warnings.append(
        f"perf retry budget exhausted: final speedup {current_speedup:.3f} below target {target:.3f}"
    )
```

New:
```python
if best_speedup < target:
    warnings.append(
        f"perf retry budget exhausted: best speedup {best_speedup:.3f} below target {target:.3f}"
    )
```

**Update the return statement** (currently around line 277):

Current:
```python
return current_artifact, current_benchmark, current_speedup, warnings, notes
```

New:
```python
return best_artifact, best_benchmark, best_speedup, warnings, notes
```

- [ ] **Step 2: Run all 3 new tests + the 2 updated existing tests**

```
python -m pytest tests/unit/stages/test_performance.py::test_perf_retry_loop_exhausts_budget_past_target tests/unit/stages/test_performance.py::test_perf_retry_loop_returns_best_so_far_on_regression tests/unit/stages/test_performance.py::test_perf_retry_loop_first_wins_on_tie tests/unit/stages/test_performance.py::test_stage4_performance_retries_until_target_met tests/unit/stages/test_performance.py::test_stage4_performance_records_failed_compile_in_warnings_and_continues -v
```

Expected: all 5 PASS.

- [ ] **Step 3: Run the entire performance test file to catch unintended breakage**

```
python -m pytest tests/unit/stages/test_performance.py -v
```

Expected: all 14 tests PASS (11 existing + 3 new).

If any existing test breaks beyond the two we proactively updated, STOP and investigate — diagnose via @superpowers:systematic-debugging. Likely culprits:
- A test that relied on `current_*` being the LAST attempt rather than the BEST (unlikely; "last == best" when speedup monotonically improves, which is the typical mock pattern).
- A test whose terminal-warning string check uses the old `"final speedup"` text.

- [ ] **Step 4: Run full unit suite to confirm no cross-cutting damage**

```
python -m pytest --ignore=tests/integration
```

Expected: 129 passed (126 baseline + 3 new), 1 warning.

- [ ] **Step 5: Lint + types**

```
python -m ruff check src tests
python -m mypy src
```

Expected: clean for both.

- [ ] **Step 6: Commit**

```bash
git add src/cuda_engine/stages/performance.py tests/unit/stages/test_performance.py
git commit -m "$(cat <<'EOF'
feat(stage4): best-so-far tracking + exhaust budget past parity

The perf retry loop used to early-exit the moment current_speedup >= target,
which capped 6+ plateau-cluster kernels at exactly 1.00× because they hit
parity on attempt 1 and the loop walked away. Now the loop always exhausts
its budget, tracks best-so-far separately from current (feeds LLM the
latest attempt for continuity, returns the best at end), and uses strict
> comparison so ties don't replace winners.

Combined with the perf_fix prompt nudge (next commit), targets lifting
fast_1 from 1/30 toward the M3 ≥10/30 gate.

Two existing tests had retry_budget adjusted (3→1 and 3→2) to align with
the new no-early-exit semantics; intent preserved.
EOF
)"
```

---

## Chunk 2: perf_fix prompt — beat-torch-compile levers

Append the new guidance block to the prompt file and lock it with a prompt-content assertion test.

### Task 2.1: Add prompt-content assertion test (RED)

**Files:**
- Modify: `tests/unit/test_prompts.py` (append)

- [ ] **Step 1: Append failing test**

Append to `tests/unit/test_prompts.py`:

```python
def test_load_prompt_perf_fix_includes_beat_torch_compile_guidance() -> None:
    """perf_fix gives the LLM concrete levers to push past 1.0× parity."""
    prompt = load_prompt("perf_fix")

    assert "Matching torch.compile is acceptable but not the goal" in prompt
    assert "float4" in prompt
    assert "108 SMs" in prompt
    assert "#pragma unroll" in prompt
    assert "__shfl_down_sync" in prompt
```

- [ ] **Step 2: Run to confirm RED**

```
python -m pytest tests/unit/test_prompts.py::test_load_prompt_perf_fix_includes_beat_torch_compile_guidance -v
```

Expected: FAIL — the new phrases don't yet exist in `perf_fix.md`. (Note: `__shfl_down_sync` may already be present from the existing "Shared-memory tiling" bullet; that's fine, the assertion still verifies the prompt contains it.)

### Task 2.2: Append the prompt block

**Files:**
- Modify: `src/cuda_engine/prompts/perf_fix.md`

- [ ] **Step 1: Read the current prompt to find the insertion point**

The new block goes AFTER the "Optimization themes to consider" bullet list ends and BEFORE the "Output the complete revised CUDA source..." closing instruction. Currently in `perf_fix.md`, the last bullet of the themes list reads:

```
- **Simple fused elementwise kernels**: for one-pass pointwise or fused
  pointwise work, prefer one coalesced read/compute/write pass with enough
  blocks to cover the tensor. Do not add multi-pass reductions, shared-memory
  staging, or complicated synchronization unless the KernelSpec actually
  requires cross-element communication.
```

And the next line is:

```
Output the complete revised CUDA source as one fenced `cuda` code block,
```

Insert the new block between them, with one blank line above and below.

- [ ] **Step 2: Insert the new block**

Insert this exact content (no leading blank inside, one blank line before and after the block):

```markdown
Matching torch.compile is acceptable but not the goal. To strictly beat it on A100:
- Prefer vectorized memory ops: `float4` for fp32, `__half2` for fp16. They double effective bandwidth on aligned contiguous data.
- Align grid to A100's 108 SMs. A full wave is a multiple of 108 blocks; a partial tail wave wastes runtime. For tensors that don't divide evenly, prefer fewer-larger blocks over more-smaller.
- Maximize instruction-level parallelism: `#pragma unroll` inner loops with small bounded trip count. Keep enough independent work per thread to hide arithmetic and memory latency.
- Fuse passes when the KernelSpec permits. Reductions followed by elementwise can often be one-pass with `__shfl_down_sync` warp reductions.
- Inspect register pressure first if Nsight shows occupancy < 50%. If regs/thread > 64 on a 256-thread block, work-split or block-size reduction frees waves.
```

- [ ] **Step 3: Run the prompt test to confirm GREEN**

```
python -m pytest tests/unit/test_prompts.py::test_load_prompt_perf_fix_includes_beat_torch_compile_guidance -v
```

Expected: PASS.

- [ ] **Step 4: Run the full prompts test file to confirm no regression**

```
python -m pytest tests/unit/test_prompts.py -v
```

Expected: all 5 PASS (4 existing + 1 new).

- [ ] **Step 5: Run full unit suite**

```
python -m pytest --ignore=tests/integration
```

Expected: 130 passed (129 + 1 new prompt test).

- [ ] **Step 6: Lint + types**

```
python -m ruff check src tests
python -m mypy src
```

Expected: clean for both.

- [ ] **Step 7: Commit**

```bash
git add src/cuda_engine/prompts/perf_fix.md tests/unit/test_prompts.py
git commit -m "$(cat <<'EOF'
feat(prompt): beat-torch-compile levers in perf_fix

Adds a "Matching torch.compile is acceptable but not the goal" block
to the perf_fix prompt with five concrete A100 levers: vectorized
memory ops (float4/__half2), 108-SM wave alignment, ILP via
#pragma unroll, fused passes with warp reductions, and register
pressure inspection.

Pairs with the loop fix in the previous commit. Without the loop fix
the LLM gets the nudge but the early-exit prevents acting on it; with
just the loop fix the LLM iterates without new direction.
EOF
)"
```

---

## Chunk 3: Colab eval validation (manual checkpoint)

The unit tests prove the mechanics. Real fast_1 lift only shows up on a Colab A100 eval rerun.

### Task 3.1: Push branch and run eval on Colab

This task is not automatable.

- [ ] **Step 1: Push the branch**

```bash
git push origin m3/perf-loop
```

- [ ] **Step 2: On Colab Pro (A100), with ANTHROPIC_API_KEY restored:**

```python
%cd /content
!rm -rf Cuda-Engine
!git clone --branch m3/perf-loop --depth 5 https://github.com/shivnarainms22/Cuda-Engine.git
%cd Cuda-Engine
!pip install -e . --quiet

import os
from google.colab import userdata
os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")

OUT_DIR = "/content/drive/MyDrive/cuda-engine-evals/2026-05-11-fast1-lift"
!mkdir -p {OUT_DIR}
!cuda-engine eval --suite internal --out {OUT_DIR} --resume 2>&1 | tee {OUT_DIR}/run.log
```

This is a fresh run, not a resume of the previous Drive directory — we want the new loop + prompt behavior applied to all 30 kernels.

- [ ] **Step 3: Read the summary**

```bash
!cat {OUT_DIR}/summary.md
```

Look for the M3 Metrics block:
```
- fast_1 kernels (>1.0x): N/30
- Below target kernels: M/30
- Pass rate: K/30
```

- [ ] **Step 4: Interpret**

| `fast_1` | Verdict |
|---|---|
| ≥ 10/30 | ✅ M3 fast_1 gate closed. Proceed to update memory + decide on merge to main. |
| 5 ≤ fast_1 < 10 | ⚠️ Change moves the needle but doesn't close the gate. Open follow-up brainstorm for Approach C (dynamic prompt branching) or the stuck-below cluster. |
| < 5 | ❌ Change is not enough. Likely needs Approach C, deeper Nsight feedback, or revisiting the loop behavior. Don't merge to main yet. |

Also check pass_rate regression: if `pass_rate < 28/30` (the baseline), some change broke a previously-passing kernel — investigate before declaring done.

### Task 3.2: Append evidence to `M3-evidence.md`

**Files:**
- Modify: `docs/milestones/M3-evidence.md` (append a new `## fast_1 Lift Checkpoint` section)

- [ ] **Step 1: Append the new section**

Pattern after the existing checkpoint sections. Include:
- Run date, environment, branch, commit SHA tested.
- Durable output directory + zip path (if a zip was created).
- The `--out` flag value.
- The full `summary.md` content (paste verbatim).
- Per-kernel speedups for any kernel that changed status from previous run (was 1.00 → now 1.05, etc.).
- Verdict (one of the three from Task 3.1 Step 4).
- Next-step recommendation.

- [ ] **Step 2: Commit**

```bash
git add docs/milestones/M3-evidence.md
git commit -m "docs(m3): post-fast1-lift eval evidence"
git push origin m3/perf-loop
```

### Task 3.3: Update memory

**Files:**
- Modify: `C:\Users\Shivnarain\.claude\projects\D--Cuda-Engine\memory\project_cuda_engine.md`

- [ ] **Step 1: Update the project memory entry**

Add a line noting:
- fast_1 lift implemented and measured: `fast_1=N/30` (date).
- Whether M3 fast_1 gate is closed.
- Next planned task (M3 closeout vs. M4 kickoff vs. stuck-below brainstorm).

---

## Final verification

After all three chunks complete:

- [ ] **Step 1: Full unit suite green**

```
python -m pytest --ignore=tests/integration
```

Expected: 130 passed.

- [ ] **Step 2: Lint + types clean**

```
python -m ruff check src tests
python -m mypy src
```

Expected: clean for both.

- [ ] **Step 3: Commit count check**

```
git log --oneline 319c468..HEAD
```

Expected: 3 commits (1 loop change, 1 prompt change, 1 evidence after Colab).

- [ ] **Step 4: Colab eval evidence appended to `M3-evidence.md`** (Task 3.2 done).

- [ ] **Step 5: Memory updated** (Task 3.3 done).

---

## Open questions (resolve during execution)

- **`notes` format string** — the plan uses `max(best_speedup, new_speedup):.3f` to compute the "best" value at the moment of the note (before `best_speedup` itself is mutated). That's correct under strict-> tie semantics but readable code might prefer computing `next_best = max(best_speedup, new_speedup)` once before the note. Implementer's discretion; both produce identical output.
- **Cost increase per Stage 4 invocation** — the change forces ALL retries to run on plateau-hitters rather than early-exiting after attempt 1. For the v1 budget (`retry_budgets.performance=3`), this triples Stage 4 LLM cost on those kernels. Acceptable per design doc section 10. If post-eval shows the cost-per-fast_1-win is too high, revisit by reducing default `retry_budgets.performance` to 2.
