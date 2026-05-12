# torch.compile Baseline Measurement — Design Document

| Field | Value |
|---|---|
| **Status** | Approved (2026-05-11) |
| **Owner** | Shivnarain |
| **Branch** | `m3/perf-loop` |
| **Parent design** | `docs/superpowers/specs/2026-04-26-cuda-synthesis-engine-design.md` |
| **Implementation plan** | `docs/superpowers/plans/2026-05-11-torch-compile-baseline-plan.md` (to be created) |
| **Milestone** | M3 — fast_1 measurement infrastructure |

---

## 1. Background and goal

The M3 eval reports `fast_1 = K/30` where fast_1 = "speedup > 1.0× vs `torch.compile`". On 2026-05-11 we discovered that the function computing this baseline (`_can_vector_add_baseline` in `src/cuda_engine/services/gpu/_run_kernel_child.py`) is a stub: it only fires when the kernel takes exactly 2 same-shape inputs, and it times `inputs[0] + inputs[1]` (a plain CUDA element-wise add) — not `torch.compile` of the reference. For every other kernel `baseline_ms` is `None`, and `_speedup()` returns `1.0` as a default, silently mislabeling unmeasured kernels as "matches parity."

Concretely, in our most recent 12-kernel rerun:
- 10/12 kernels had `baseline_ms = None`
- All 10 reported `speedup_vs_torch_compile = 1.00`
- Only the 2 kernels matching the vector-add stub got real numbers

Every fast_1 number we have measures "did the LLM beat a plain CUDA add" on kernels that happen to be 2-input-same-shape. It does not measure what M3 promised.

**Goal:** replace the stub with a real `torch.compile`-based baseline, surface measurement failures explicitly (no silent default), and update the eval runner so fast_1 is honest.

---

## 2. Scope

### In scope
- New `reference: Callable[..., Any] | None = None` parameter on `GPURunner.benchmark_kernel` and `LocalGPURunner.benchmark_kernel`.
- Subprocess child (`_run_kernel_child.py`) gains a real `torch.compile`-based baseline routine. Replaces `_can_vector_add_baseline` + the `inputs[0] + inputs[1]` baseline body.
- Reference function pickled into the child's input payload.
- `_speedup()` returns `None` (not `1.0`) when `baseline_ms is None` or `custom_ms <= 0`.
- `PerformanceReport.speedup_vs_torch_compile` becomes `float | None`.
- `Stage4Performance.run()` threads the reference into `benchmark_kernel`; the retry loop measures baseline once and reuses it.
- Orchestrator threads `reference` to Stage 4 (Stage 4 doesn't currently receive it).
- Eval runner adds `baseline_status` column to `results.csv`; `summary.md` separates `fast_1` from `baseline_failed`.
- Unit tests using `MockGPURunner` cover the new `reference` parameter contract and the `None`-speedup semantics.
- One integration test confirms `torch.compile` baseline works against real CUDA.

### Out of scope
- Persistent baseline cache across eval runs. The retry loop already reuses baseline within one Stage 4 invocation; cross-run caching is YAGNI for v1.
- Multiple `torch.compile` modes as a config knob. We pick `mode="reduce-overhead"` and document the choice. Configurable later if real evidence demands.
- Re-running Stage 4 on kernels that previously hit fast_1 with the stub baseline — those numbers were measured against vector-add, not torch.compile, and will likely look different under the real baseline. Acceptable; the post-fix run is the new truth.
- Closures, lambdas, partials with closed-over GPU tensors as references. Public API now documents that references must be top-level functions or `functools.partial` of them. Failing references surface as `baseline_failed` with the pickle error.

### Explicitly rejected
- **Parent-process baseline measurement.** Breaks the M1 invariant that all GPU work happens in the subprocess. `torch.compile` is exactly the kind of code that can produce nasty crashes (graph capture failures, memory corruption) — keeping it in the isolated child is the architectural call.
- **Eager-mode baseline only.** Beats the M3 contract. fast_1 needs to mean "beat torch.compile" or the gate is meaningless.
- **Treating `baseline_failed` as fast_1=0 in the metric.** It would let measurement failures pass for "we got close." Honest accounting: failures don't count as wins, they count as failures, and we surface them.

---

## 3. Architecture

```
synthesize(reference=fn, ...)
       │
       ▼
Orchestrator.run(prompt, reference, target)
       │
       │ stages 1, 3, 5 already receive reference today
       ▼
Stage4Performance.run(spec, artifact, run_id, retry_budget, reference)   ◄── NEW arg
       │
       │ measures baseline once at first benchmark, reuses for retry loop
       ▼
LocalGPURunner.benchmark_kernel(so, inputs, reference=fn, ...)   ◄── NEW arg
       │
       │ pickles (so_path, inputs, reference) into subprocess payload
       ▼
_run_kernel_child.py
   _benchmark_forward(...)
       ├── time custom kernel (existing)
       └── if reference is not None:
              _measure_torch_compile_baseline(torch, reference, inputs, …)
                 try:
                     compiled = torch.compile(reference, mode="reduce-overhead")
                     warmup × N
                     synchronize
                     time × M iterations  → baseline_ms
                 except Exception as e:
                     baseline_ms = None
                     baseline_error = "torch.compile baseline failed: …"
       └── returns BenchmarkResult{custom_ms, baseline_ms, baseline_error, …}
```

**Failure propagation:**
```
baseline_failed in child  →  BenchmarkResult.baseline_ms=None, BenchmarkResult.stderr=<error>
                          →  Stage 4 _speedup() returns None
                          →  PerformanceReport.speedup_vs_torch_compile=None, below_target=True
                          →  warnings include "torch.compile baseline failed: …"
                          →  Eval row: baseline_status="failed", failure_kind="baseline_failed"
                          →  Summary md: counted in "baseline_failed: M/30", NOT in fast_1
```

---

## 4. Components

### 4.1 `GPURunner.benchmark_kernel` ABC signature

```python
def benchmark_kernel(
    self,
    so_path: Path,
    inputs: list[Any],
    *,
    reference: Callable[..., Any] | None = None,   # NEW
    warmup_iterations: int = 10,
    timed_iterations: int = 50,
    timeout_seconds: int = 60,
) -> BenchmarkResult: ...
```

`reference=None` preserves backward compat for callers without a reference (unit tests, future API consumers). Default keeps the signature additive.

### 4.2 `BenchmarkResult` schema

```python
class BenchmarkResult(BaseModel):
    ok: bool
    custom_ms: float = 0.0
    baseline_ms: float | None = None              # unchanged type
    baseline_error: str | None = None             # NEW — surfaces baseline failure reason
    achieved_gbps: float | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    warmup_iterations: int = 0
    timed_iterations: int = 0
```

`baseline_error` is set only when the baseline measurement was attempted and failed. `None` means either "succeeded" (`baseline_ms` is populated) or "not attempted" (`reference` was `None`).

### 4.3 `LocalGPURunner.benchmark_kernel` pickle payload

The subprocess input pickle gains a `reference` entry. Current payload is `inputs` (a list); new payload is a dict so we can add fields without further breakage:

```python
payload = {
    "inputs": inputs,
    "reference": reference,   # None if not provided
}
pickle.dump(payload, input_file)
```

Child's argparse path stays — only the pickle contents change. Backward-compat: if a payload comes through as a bare list (older fixture, in-tree compat), the child falls back to `{"inputs": payload, "reference": None}`.

### 4.4 `_run_kernel_child.py` baseline routine

Replace `_can_vector_add_baseline` + the inline baseline measurement with:

```python
def _measure_torch_compile_baseline(
    torch: Any,
    reference: Any,
    inputs: list[Any],
    *,
    warmup_iterations: int,
    timed_iterations: int,
) -> tuple[float | None, str | None]:
    """Returns (baseline_ms, error_str). Exactly one of them is None."""
    try:
        compiled = torch.compile(reference, mode="reduce-overhead")
        for _ in range(warmup_iterations):
            compiled(*inputs)
        _synchronize_if_cuda(torch, inputs)
        baseline_ms = _time_callable_ms(
            torch, lambda: compiled(*inputs),
            iterations=timed_iterations,
            use_cuda_events=_has_cuda_inputs(inputs),
        )
        return baseline_ms, None
    except Exception as exc:
        return None, f"torch.compile baseline failed: {type(exc).__name__}: {exc}"
```

Called from `_benchmark_forward` after `custom_ms` is computed. If `reference is None`, both return values are `None` and `baseline_error` stays `None` in the result (skipped, not failed).

### 4.5 `_speedup` no longer defaults to 1.0

```python
# src/cuda_engine/stages/performance.py
def _speedup(*, baseline_ms: float | None, custom_ms: float) -> float | None:
    if custom_ms <= 0 or baseline_ms is None:
        return None
    return baseline_ms / custom_ms
```

Callers must handle `None`. Three sites today: initial speedup computation, retry loop's new-speedup, and the `current_speedup < target` comparison (now `current_speedup is not None and current_speedup < target` — `None` is treated as "below target").

### 4.6 `PerformanceReport.speedup_vs_torch_compile`

```python
class PerformanceReport(BaseModel):
    speedup_vs_torch_compile: float | None = None   # was: float, defaulted 0.0
    speedup_vs_reference: float | None = None       # same change for symmetry
    below_target: bool
    achieved_gbps: float | None = None
    warnings: list[str] = []
    notes: list[str] = []
```

`None` means "could not be measured." `below_target` remains `bool` and is `True` when speedup is `None` OR `< target`. The CLI and eval rendering format `None` as `"n/a"`.

### 4.7 `Stage4Performance.run` reference threading + baseline reuse

```python
def run(
    self, *, spec, artifact, run_id, retry_budget=3, reference: Callable[..., Any] | None = None,
):
    ...
    benchmark = self.gpu.benchmark_kernel(so, inputs, reference=reference, ...)
    cached_baseline = benchmark.baseline_ms              # measure once
    speedup = _speedup(baseline_ms=cached_baseline, custom_ms=benchmark.custom_ms)
    ...
    if self.llm is not None and retry_budget > 0:
        # retry loop receives cached_baseline; doesn't re-measure
        ... self._retry_loop(..., baseline_ms=cached_baseline)
```

Inside `_retry_loop`, each retry's `benchmark_kernel` call now passes `reference=None` (we've already got the baseline) and `_speedup` uses the cached value:

```python
new_benchmark = self.gpu.benchmark_kernel(so, inputs, reference=None, ...)
# new_benchmark.baseline_ms will be None (we didn't pass reference)
new_speedup = _speedup(baseline_ms=cached_baseline, custom_ms=new_benchmark.custom_ms)
```

This saves N-1 redundant torch.compile invocations per kernel.

### 4.8 Orchestrator threading

`Orchestrator.run` line ~149 (the Stage 4 invocation) gains `reference=reference`:

```python
performance, artifact = _run_traced_stage(
    stage_traces, llm, "performance",
    lambda: Stage4Performance(...).run(
        spec=spec, artifact=artifact, run_id=run_id,
        retry_budget=self.cfg.retry_budgets.performance,
        reference=reference,                                # NEW
    ),
)
```

### 4.9 Eval runner CSV + summary

`evals/runner.py`:
- New CSV column `baseline_status` with values `ok` / `failed` / `skipped`.
  - `ok`: baseline_ms populated
  - `failed`: baseline_error set
  - `skipped`: reference was None (shouldn't happen in eval runs but possible if a kernel fixture is broken)
- `failure_kind` gains a new value `baseline_failed` (matches the existing `runner_error` / `correctness_failed` / `compile_failed` enum).
- `summary.md` "M3 Metrics" block separates:
  ```
  - Pass rate (with measurable baseline): N/30
  - fast_1 kernels (>1.0x with measured baseline): K/30
  - baseline_failed: B/30      ← NEW
  - below_target (with measured baseline): P/30
  ```

The fast_1 denominator stays 30 — kernels with failed baselines don't count as wins. Honest.

---

## 5. Data flow — full path of one Stage 4 invocation

```
1. synthesize(prompt, reference=rms_norm_fn, target="sm_80") called from eval runner
2. Orchestrator passes reference to Stage 1, 3, 4, 5
3. Stage 4 calls gpu.benchmark_kernel(so_path, inputs, reference=rms_norm_fn)
4. LocalGPURunner pickles {"inputs": [x], "reference": rms_norm_fn} into temp file
5. Subprocess spawned with --so, --input, --output paths
6. Child unpickles, sees reference is set
7. Child times custom kernel via _time_callable_ms → custom_ms
8. Child runs _measure_torch_compile_baseline:
     - torch.compile(rms_norm_fn, mode="reduce-overhead")
     - 10 warmup iterations
     - synchronize
     - 100 timed iterations → baseline_ms
9. Child writes BenchmarkResult{custom_ms=0.0099, baseline_ms=0.0083, ...} as pickle
10. Parent reads back, returns BenchmarkResult
11. Stage 4: speedup = _speedup(baseline_ms=0.0083, custom_ms=0.0099) = 0.84
12. cached_baseline = 0.0083 stored for retry loop
13. PerformanceReport.speedup_vs_torch_compile = 0.84
14. below_target = True (0.84 < 1.0)
15. Stage 4 enters retry loop, each retry's _speedup uses cached_baseline (no recompile)
16. Final returned speedup is best across attempts, measured against the cached 0.0083
17. Eval runner records: speedup_vs_torch_compile=0.84, baseline_status="ok"
18. Summary tallies: below_target counts this kernel; fast_1 does not
```

For a failing baseline:
```
8'. torch.compile raises (e.g., dynamic shape it can't handle)
9'. Child returns baseline_ms=None, baseline_error="torch.compile baseline failed: ..."
10'. Parent reads back
11'. _speedup returns None
12'. PerformanceReport.speedup_vs_torch_compile = None, below_target = True
12''. warnings += ["torch.compile baseline failed: ..."]
13'. Stage 4 retry loop: cached_baseline is None, so all retry _speedup calls return None too
14'. Final report has speedup=None, below_target=True
15'. Eval row: baseline_status="failed", failure_kind="baseline_failed"
16'. Summary: counted in baseline_failed bucket, not in fast_1
```

---

## 6. Error handling

| Scenario | Behavior |
|---|---|
| `reference=None` (no reference provided) | `baseline_ms=None`, `baseline_error=None`. `speedup=None`. Below-target. Eval row: `baseline_status="skipped"`. |
| Reference is not pickleable (lambda, closure) | Pickle raises in parent before subprocess spawn. Caught, reported as `baseline_error="reference not pickleable: ..."`. Result identical to above. |
| `torch.compile(reference)` raises in child | Caught, `baseline_error` populated. Custom kernel still benchmarked normally. |
| `compiled(*inputs)` raises during warmup or timing | Same as above. We DO NOT swallow custom kernel benchmark failures — those still propagate as `ok=False`. |
| Subprocess timeout | Existing handling: `timed_out=True`, custom_ms=0, baseline_ms=None. |
| Reference produces different output shape than custom kernel | Not our problem — correctness is Stage 3's job. Stage 4 only times. |
| All retries' baselines fail consistently | Each retry uses cached `None` baseline. Speedups stay `None`. Behavior is identical to "single failed measurement"; no infinite loop. |
| Eval runner reads `speedup_vs_torch_compile=None` from a per-kernel JSON | CSV cell is empty string; markdown table cell is `n/a`. No KeyError. |

---

## 7. Tests

### Unit tests

**`tests/unit/services/gpu/test_mocks.py`** — extend MockGPURunner:
- `test_mock_gpu_runner_records_reference_param` — verify `benchmark_kernel(reference=fn)` is captured in `benchmark_calls`.
- `test_mock_gpu_runner_baseline_passthrough` — assert queued BenchmarkResults with `baseline_ms=None, baseline_error="..."` flow through unchanged.

**`tests/unit/stages/test_performance.py`** — extend Stage 4 tests:
- `test_speedup_returns_none_when_baseline_missing` — direct unit on `_speedup`.
- `test_speedup_returns_none_when_custom_ms_zero` — edge case.
- `test_stage4_below_target_when_baseline_failed` — fixture: `BenchmarkResult(ok=True, custom_ms=0.01, baseline_ms=None, baseline_error="...")`. Assert `report.speedup_vs_torch_compile is None`, `report.below_target is True`, `"torch.compile baseline failed" in report.warnings`.
- `test_stage4_retry_loop_reuses_cached_baseline` — verify that after initial benchmark with reference, subsequent retry's `benchmark_kernel` calls have `reference=None` (we don't re-measure).

**`tests/unit/test_orchestrator.py`** — extend:
- `test_orchestrator_threads_reference_to_stage4` — assert `MockGPURunner.benchmark_calls[0]["reference"]` is the reference function passed to `Orchestrator.run`.

**`tests/unit/test_models.py`** (or wherever PerformanceReport tests live) — verify `speedup_vs_torch_compile: float | None`.

### Integration test

**`tests/integration/test_baseline_torch_compile.py`** — gated on CUDA + nvcc:
- Top-level def `_relu_reference(x): return torch.relu(x)` (so it pickles).
- Build a trivial CUDA relu kernel (or reuse vector_add fixture).
- Call `LocalGPURunner.benchmark_kernel(so, [x], reference=_relu_reference)`.
- Assert `baseline_ms` is `float`, between 1e-3 and 100 (sane range for a tensor of moderate size).
- Assert `baseline_error is None`.
- Negative case: pass a deliberately-broken reference that raises in `torch.compile`. Assert `baseline_ms is None, baseline_error is not None`, but `custom_ms` is still populated.

### What we DON'T test
- `torch.compile` internals. That's PyTorch's job.
- That `torch.compile` is actually faster than eager — depends on hardware and kernel shape, not deterministic.

---

## 8. Eval validation (post-implementation)

After commits land, run on Colab:
```
cuda-engine eval --suite internal --out <new-dir>
```

Expectations:
- Every kernel reports a real `speedup_vs_torch_compile` OR `baseline_status="failed"` with an error message.
- The 10 plateau kernels from 2026-05-11-fast1-lift-v2 (layernorm, rms_norm, softmaxes, etc.) now show real speedups — likely below 1.0× given torch.compile's strong elementwise/reduction performance.
- True fast_1 number is honestly measured for the first time.
- We can then make an informed decision about whether to:
  - Accept current fast_1 as the M3 result (e.g., "M3 closes at fast_1=3/30 with honest measurement").
  - Invest in further LLM optimization (different prompt, multi-attempt Opus, etc.) now that we have real signal.
  - Pivot fast_1 target downward in the M3 charter if torch.compile is structurally hard to beat.

---

## 9. Commit plan (preview)

1. `feat(gpu): add reference param to benchmark_kernel + baseline_error to BenchmarkResult` — ABC + Mock + Local signature updates, no behavior change yet.
2. `feat(gpu): real torch.compile baseline in subprocess child` — implement `_measure_torch_compile_baseline`, update pickle payload format, remove `_can_vector_add_baseline` stub.
3. `feat(stage4): make _speedup return None on missing baseline, thread reference` — surface failures explicitly. Update `PerformanceReport` schema. Cache baseline in retry loop.
4. `feat(orchestrator): pass reference to Stage 4` — one-line thread through.
5. `feat(evals): baseline_status column + honest fast_1 summary` — CSV + markdown changes.
6. `test(integration): torch.compile baseline against real GPU` — Colab integration test.
7. `docs(m3): record measurement-bug discovery + corrected baseline run` — append to M3-evidence.md after Colab confirms.

~7 commits, ~300–400 LOC + tests.

---

## 10. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `mode="reduce-overhead"` doesn't apply to small tensors and benchmark dominates by CUDA graph overhead | Medium | Document; can fall back to `mode="default"` per-kernel later if real data shows this. Not blocking initial implementation. |
| Pickling reference function fails for some eval kernel | Low | Eval kernels are top-level functions, verified pickleable. Failures surface as `baseline_failed` with the pickle error. |
| Existing tests break due to `_speedup` signature change | Medium | Plan covers every call site explicitly. Tests fixture-update upfront. |
| `torch.compile` first-call cost dominates wall time | Medium-low | Warmup iterations exist for this exact reason. `reduce-overhead` mode is fastest. |
| Post-fix fast_1 is much lower than before (e.g., 0-2/30 instead of 4-5/30) | High | This is the honest result. M3 charter may need adjustment. Better to know now than ship a meaningless metric. |
| Reference functions in `evals/internal/*/reference.py` use ops `torch.compile` can't trace | Medium | Surfaces as `baseline_failed`. We fix or annotate per kernel. |
| Real-API integration test is flaky | Low | Test marked `@pytest.mark.integration`, skipped on non-CUDA hosts, run manually on Colab. |

---

## 11. Open questions

None at design time. Surfaces during execution get logged in the plan.
