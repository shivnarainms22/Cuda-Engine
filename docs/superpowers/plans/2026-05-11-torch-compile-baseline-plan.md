# torch.compile Baseline Measurement Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the vector-add stub baseline with a real `torch.compile`-based baseline measurement so the M3 `fast_1` metric becomes truthful. Surface measurement failures explicitly (no silent `1.0` fallback).

**Architecture:** Pass the reference function through `synthesize() → Orchestrator → Stage 4 → GPURunner.benchmark_kernel → subprocess child`. Child pickles the reference and calls `torch.compile(reference, mode="reduce-overhead")`, warms up, times. Failures populate `BenchmarkResult.baseline_error` and propagate as `speedup_vs_torch_compile = None`, `below_target = True`, with a clear warning. Retry loop measures baseline once and reuses it across attempts.

**Tech Stack:** Python 3.11+, pytest, Pydantic v2, PyTorch ≥2.4 with `torch.compile`, existing `MockLLMClient`/`MockGPURunner` for unit tests.

**Spec:** `docs/superpowers/specs/2026-05-11-torch-compile-baseline-design.md`

**Skills referenced:**
- @superpowers:test-driven-development — every code task: failing test first.
- @superpowers:verification-before-completion — evidence-only completion claims.
- @superpowers:systematic-debugging — when a step fails, diagnose root cause.

**Branch:** `m3/perf-loop` (HEAD `f8b97de` after spec commit). Baseline: 132 unit tests passing, ruff clean, mypy clean.

---

## File Structure (locked)

### Files modified
| Path | Change |
|---|---|
| `src/cuda_engine/services/gpu/base.py` | `BenchmarkResult.baseline_error: str \| None = None`; `GPURunner.benchmark_kernel` adds `reference: Callable[..., Any] \| None = None` kwarg |
| `src/cuda_engine/services/gpu/mocks.py` | `MockGPURunner.benchmark_kernel` accepts `reference`, records in `benchmark_calls` |
| `src/cuda_engine/services/gpu/local.py` | Pickle payload becomes `{"inputs": ..., "reference": ...}`; pass `reference` arg through |
| `src/cuda_engine/services/gpu/_run_kernel_child.py` | Backward-compat pickle read; replace `_can_vector_add_baseline` + body with `_measure_torch_compile_baseline`; return `baseline_error` |
| `src/cuda_engine/models/reports.py` | `PerformanceReport.speedup_vs_torch_compile: float \| None`, `speedup_vs_reference: float \| None` |
| `src/cuda_engine/stages/performance.py` | `_speedup` returns `Optional[float]`; `Stage4Performance.run()` accepts `reference`, caches baseline; retry loop reuses cached baseline; `below_target` computed from `None`-or-`<target` |
| `src/cuda_engine/orchestrator.py` | Thread `reference` to Stage 4 |
| `evals/runner.py` | Add `baseline_status` to `CSV_COLUMNS` + `EvalRow`; classify `baseline_failed`; summary md splits fast_1 from baseline_failed |
| Multiple test files | Update fixtures to handle `float \| None` speedups; add baseline-failure-path tests |

### Files created
| Path | Purpose |
|---|---|
| `tests/integration/test_baseline_torch_compile.py` | Real CUDA + torch.compile integration test |

---

## Chunk 1: BenchmarkResult schema + GPURunner ABC

Additive signature change. No behavior change yet. Lands first so every downstream chunk can pin against the new shape.

### Task 1.1: Add `baseline_error` field to `BenchmarkResult` (RED)

**Files:**
- Modify: `src/cuda_engine/services/gpu/base.py`
- Modify: `tests/unit/services/test_abcs_exist.py` (or whichever test asserts the schema)

- [ ] **Step 1: Locate or write the failing test**

Search for existing `BenchmarkResult` schema tests:
```
grep -n "baseline_error\|baseline_ms" tests/unit/services/
```

If a schema test exists, extend it. Otherwise append to `tests/unit/services/test_abcs_exist.py`:

```python
def test_benchmark_result_carries_baseline_error_field() -> None:
    """Failed torch.compile baseline must surface as a structured error string."""
    from cuda_engine.services.gpu.base import BenchmarkResult

    result = BenchmarkResult(
        ok=True,
        custom_ms=0.01,
        baseline_ms=None,
        baseline_error="torch.compile baseline failed: RuntimeError: ...",
    )

    assert result.baseline_error == "torch.compile baseline failed: RuntimeError: ..."
    # Default for unspecified
    assert BenchmarkResult(ok=True, custom_ms=0.01).baseline_error is None
```

- [ ] **Step 2: Run to confirm RED**

```
python -m pytest tests/unit/services/test_abcs_exist.py::test_benchmark_result_carries_baseline_error_field -v
```

Expected: FAIL with `ValidationError` (unknown field) or `AttributeError`.

- [ ] **Step 3: Add the field to `BenchmarkResult`**

In `src/cuda_engine/services/gpu/base.py`:

```python
class BenchmarkResult(BaseModel):
    ok: bool
    custom_ms: float = 0.0
    baseline_ms: float | None = None
    baseline_error: str | None = None      # NEW
    achieved_gbps: float | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    warmup_iterations: int = 0
    timed_iterations: int = 0
```

- [ ] **Step 4: Run to confirm GREEN**

```
python -m pytest tests/unit/services/test_abcs_exist.py::test_benchmark_result_carries_baseline_error_field -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cuda_engine/services/gpu/base.py tests/unit/services/test_abcs_exist.py
git commit -m "feat(gpu): add baseline_error field to BenchmarkResult"
```

### Task 1.2: Add `reference` kwarg to `GPURunner.benchmark_kernel` ABC + Mock + Local (RED)

**Files:**
- Modify: `src/cuda_engine/services/gpu/base.py` (ABC)
- Modify: `src/cuda_engine/services/gpu/mocks.py`
- Modify: `src/cuda_engine/services/gpu/local.py`
- Modify: `tests/unit/services/test_abcs_exist.py` and/or `test_mocks.py`

- [ ] **Step 1: Write failing test for `MockGPURunner` reference passthrough**

Append to `tests/unit/services/gpu/test_mocks.py` (create if doesn't exist; check via `ls tests/unit/services/gpu/` first):

```python
from pathlib import Path

from cuda_engine.services.gpu.mocks import MockGPURunner


def _ref(x):
    return x


def test_mock_gpu_runner_records_reference_param() -> None:
    runner = MockGPURunner()
    runner.benchmark_kernel(Path("/tmp/x.so"), [object()], reference=_ref)

    call = runner.benchmark_calls[0]
    assert call["reference"] is _ref
```

- [ ] **Step 2: Run to confirm RED**

```
python -m pytest tests/unit/services/gpu/test_mocks.py::test_mock_gpu_runner_records_reference_param -v
```

Expected: FAIL with `TypeError: benchmark_kernel() got an unexpected keyword argument 'reference'`.

- [ ] **Step 3: Update ABC signature**

In `src/cuda_engine/services/gpu/base.py`, modify the `GPURunner.benchmark_kernel` abstract method:

```python
@abstractmethod
def benchmark_kernel(
    self,
    so_path: Path,
    inputs: list[Any],
    *,
    reference: Callable[..., Any] | None = None,    # NEW
    warmup_iterations: int = 10,
    timed_iterations: int = 50,
    timeout_seconds: int = 60,
) -> BenchmarkResult:
    raise NotImplementedError
```

Add `from collections.abc import Callable` to imports if missing.

- [ ] **Step 4: Update `MockGPURunner.benchmark_kernel`**

In `src/cuda_engine/services/gpu/mocks.py`, update signature:

```python
def benchmark_kernel(
    self,
    so_path: Path,
    inputs: list[Any],
    *,
    reference: Callable[..., Any] | None = None,    # NEW
    warmup_iterations: int = 10,
    timed_iterations: int = 50,
    timeout_seconds: int = 60,
) -> BenchmarkResult:
    self.benchmark_calls.append(
        {
            "so_path": so_path,
            "input_shapes": [tuple(getattr(input_value, "shape", ())) for input_value in inputs],
            "reference": reference,                     # NEW
            "warmup_iterations": warmup_iterations,
            "timed_iterations": timed_iterations,
            "timeout_seconds": timeout_seconds,
        }
    )
    # ... existing body unchanged ...
```

Add `Callable` import.

- [ ] **Step 5: Update `LocalGPURunner.benchmark_kernel` signature (no behavior change yet — just accept the arg)**

In `src/cuda_engine/services/gpu/local.py`, add `reference` to the signature. Stash it as a local `_unused_reference = reference` for now (Chunk 2 implements it):

```python
def benchmark_kernel(
    self,
    so_path: Path,
    inputs: list[Any],
    *,
    reference: Callable[..., Any] | None = None,   # NEW (unused in this chunk)
    warmup_iterations: int = 10,
    timed_iterations: int = 50,
    timeout_seconds: int = 60,
) -> BenchmarkResult:
    # Reference threading lands in Chunk 2; capture to avoid ruff unused-arg
    _ = reference
    ...
```

Add `Callable` import.

- [ ] **Step 6: Run the new test to confirm GREEN**

```
python -m pytest tests/unit/services/gpu/test_mocks.py::test_mock_gpu_runner_records_reference_param -v
```

Expected: PASS.

- [ ] **Step 7: Run full suite to confirm no regressions**

```
python -m pytest --ignore=tests/integration
```

Expected: 133 passed (132 baseline + 1 new), or 134 if Task 1.1 also added one.

- [ ] **Step 8: Lint + types**

```
python -m ruff check src tests
python -m mypy src
```

Expected: clean for both. If mypy flags `Callable` import in `base.py` or `mocks.py`, add it.

- [ ] **Step 9: Commit**

```bash
git add src/cuda_engine/services/gpu/base.py src/cuda_engine/services/gpu/mocks.py src/cuda_engine/services/gpu/local.py tests/unit/services/gpu/test_mocks.py
git commit -m "feat(gpu): add reference param to benchmark_kernel"
```

---

## Chunk 2: Real torch.compile baseline in subprocess child

Replace the vector-add stub with a real implementation. Pickle payload format changes to a dict so we can carry the reference cleanly.

### Task 2.1: Update child's pickle payload format (backward compatible)

**Files:**
- Modify: `src/cuda_engine/services/gpu/_run_kernel_child.py:20-21` (input loading)
- Modify: `src/cuda_engine/services/gpu/local.py:189` (input dumping for `run_kernel`)
- Modify: `src/cuda_engine/services/gpu/local.py:188-189` (input dumping for `benchmark_kernel`)

- [ ] **Step 1: Write a child-side test verifying it accepts both old (list) and new (dict) payload formats**

The child is invoked as a subprocess; testing it directly is awkward. Test the format-detection helper instead. Create or extend `tests/unit/services/gpu/test_run_kernel_child.py`:

```python
import pickle
from pathlib import Path

from cuda_engine.services.gpu._run_kernel_child import _load_payload  # NEW helper


def test_load_payload_accepts_legacy_list_format(tmp_path: Path) -> None:
    """Legacy: pickle payload is the inputs list directly."""
    payload_path = tmp_path / "in.pkl"
    payload_path.write_bytes(pickle.dumps([1, 2, 3]))

    inputs, reference = _load_payload(payload_path)

    assert inputs == [1, 2, 3]
    assert reference is None


def test_load_payload_accepts_new_dict_format(tmp_path: Path) -> None:
    """New: pickle payload is a dict {'inputs': [...], 'reference': fn|None}."""
    def fn(x):
        return x

    payload_path = tmp_path / "in.pkl"
    payload_path.write_bytes(pickle.dumps({"inputs": [4, 5], "reference": fn}))

    inputs, reference = _load_payload(payload_path)

    assert inputs == [4, 5]
    assert reference is fn
```

- [ ] **Step 2: Run to confirm RED**

```
python -m pytest tests/unit/services/gpu/test_run_kernel_child.py -v
```

Expected: FAIL with `ImportError: cannot import name '_load_payload'`.

- [ ] **Step 3: Implement `_load_payload` in the child**

In `src/cuda_engine/services/gpu/_run_kernel_child.py`, add the helper and rewire `main()` to use it:

```python
def _load_payload(input_path: Path) -> tuple[list[Any], Any]:
    """Load pickle payload, supporting both legacy list and new dict formats."""
    with input_path.open("rb") as f:
        raw = pickle.load(f)
    if isinstance(raw, dict):
        return raw.get("inputs", []), raw.get("reference")
    # Legacy: bare list of inputs, no reference
    return raw, None
```

Replace the current `pickle.load(f)` call at lines 20-21 of `main()` with:

```python
inputs, reference = _load_payload(Path(args.input))
```

Then `inputs` is used as before; `reference` is captured (used in Task 2.2).

- [ ] **Step 4: Update parent (`local.py`) to write the new dict format for benchmark_kernel**

In `src/cuda_engine/services/gpu/local.py`, around line 188-189 in `benchmark_kernel`:

```python
with input_path.open("wb") as f:
    pickle.dump({"inputs": inputs, "reference": reference}, f)
```

Do NOT change `run_kernel`'s payload yet — `run_kernel` doesn't need a reference. The backward-compat in `_load_payload` keeps it working.

- [ ] **Step 5: Run the new tests to confirm GREEN**

```
python -m pytest tests/unit/services/gpu/test_run_kernel_child.py -v
```

Expected: PASS.

- [ ] **Step 6: Run full suite to confirm no integration regressions**

```
python -m pytest --ignore=tests/integration
```

Expected: same count as Chunk 1, all passing.

- [ ] **Step 7: Commit**

```bash
git add src/cuda_engine/services/gpu/_run_kernel_child.py src/cuda_engine/services/gpu/local.py tests/unit/services/gpu/test_run_kernel_child.py
git commit -m "feat(gpu): pickle payload supports reference function alongside inputs"
```

### Task 2.2: Implement `_measure_torch_compile_baseline` in the child

**Files:**
- Modify: `src/cuda_engine/services/gpu/_run_kernel_child.py` — replace `_can_vector_add_baseline` + the baseline body in `_benchmark_forward`

- [ ] **Step 1: Write failing test for the new baseline helper**

This is GPU-bound; unit-testing it without CUDA requires either mocking torch (fragile) or skipping. The right test is integration (Chunk 5). For the unit, we test the **failure path** — passing a reference that raises:

Append to `tests/unit/services/gpu/test_run_kernel_child.py`:

```python
def test_measure_torch_compile_baseline_captures_failure(monkeypatch) -> None:
    """When torch.compile raises, the helper returns (None, error_string)."""
    import sys, types

    from cuda_engine.services.gpu._run_kernel_child import _measure_torch_compile_baseline

    # Build a fake torch module with a torch.compile that raises
    fake_torch = types.SimpleNamespace(
        compile=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("graph capture failed")),
        cuda=types.SimpleNamespace(),
    )

    def bad_reference(x):
        return x

    baseline_ms, error = _measure_torch_compile_baseline(
        fake_torch, bad_reference, [object()],
        warmup_iterations=1, timed_iterations=1,
    )

    assert baseline_ms is None
    assert error is not None
    assert "torch.compile baseline failed" in error
    assert "RuntimeError" in error
    assert "graph capture failed" in error


def test_measure_torch_compile_baseline_returns_float_on_success(monkeypatch) -> None:
    """When torch.compile succeeds, returns (float, None)."""
    import types

    from cuda_engine.services.gpu._run_kernel_child import _measure_torch_compile_baseline

    # Fake: compile returns a function that just runs the original; cuda absent so we use perf_counter path
    fake_torch = types.SimpleNamespace(
        compile=lambda fn, **kw: fn,
        cuda=types.SimpleNamespace(),
    )

    def reference(x):
        return x  # near-instant; will benchmark to ~0ms

    baseline_ms, error = _measure_torch_compile_baseline(
        fake_torch, reference, [object()],
        warmup_iterations=1, timed_iterations=1,
    )

    assert isinstance(baseline_ms, float)
    assert error is None
```

- [ ] **Step 2: Run to confirm RED**

```
python -m pytest tests/unit/services/gpu/test_run_kernel_child.py -v
```

Expected: FAIL on the new tests (`ImportError: cannot import name '_measure_torch_compile_baseline'`).

- [ ] **Step 3: Implement the helper in the child**

In `src/cuda_engine/services/gpu/_run_kernel_child.py`, add the new helper (replace the existing `_can_vector_add_baseline`):

```python
def _measure_torch_compile_baseline(
    torch: Any,
    reference: Any,
    inputs: list[Any],
    *,
    warmup_iterations: int,
    timed_iterations: int,
) -> tuple[float | None, str | None]:
    """Time torch.compile(reference) on the given inputs.

    Returns (baseline_ms, error_str). Exactly one of them is None:
      success → (float, None)
      failure → (None, "<error message>")
    """
    try:
        compiled = torch.compile(reference, mode="reduce-overhead")
        for _ in range(warmup_iterations):
            compiled(*inputs)
        _synchronize_if_cuda(torch, inputs)
        baseline_ms = _time_callable_ms(
            torch,
            lambda: compiled(*inputs),
            iterations=timed_iterations,
            use_cuda_events=_has_cuda_inputs(inputs),
        )
        return baseline_ms, None
    except Exception as exc:
        return None, f"torch.compile baseline failed: {type(exc).__name__}: {exc}"
```

- [ ] **Step 4: Replace the baseline body in `_benchmark_forward`**

In `_benchmark_forward` (around line 100-122), replace the current `_can_vector_add_baseline` block with a call to the new helper:

```python
custom_ms = _time_callable_ms(
    torch,
    lambda: forward(*inputs),
    iterations=timed_iterations,
    use_cuda_events=_has_cuda_inputs(inputs),
)

baseline_ms: float | None = None
baseline_error: str | None = None
if reference is not None:
    baseline_ms, baseline_error = _measure_torch_compile_baseline(
        torch, reference, inputs,
        warmup_iterations=warmup_iterations,
        timed_iterations=timed_iterations,
    )

return {
    "ok": True,
    "custom_ms": custom_ms,
    "baseline_ms": baseline_ms,
    "baseline_error": baseline_error,                           # NEW
    "achieved_gbps": _achieved_gbps(inputs, custom_ms),
    "warmup_iterations": warmup_iterations,
    "timed_iterations": timed_iterations,
}
```

`reference` parameter must be threaded into `_benchmark_forward`. Update its signature and the call site in `main()`:

```python
def _benchmark_forward(
    so_path: Path,
    inputs: list[Any],
    reference: Any,                                 # NEW
    *,
    warmup_iterations: int,
    timed_iterations: int,
) -> dict[str, Any]:
```

In `main()`, pass it:

```python
benchmark = _benchmark_forward(
    Path(args.so),
    inputs,
    reference,
    warmup_iterations=args.warmup_iterations,
    timed_iterations=args.timed_iterations,
)
```

- [ ] **Step 5: Delete `_can_vector_add_baseline`**

It's dead code now. Remove the function entirely.

- [ ] **Step 6: Run new tests to confirm GREEN**

```
python -m pytest tests/unit/services/gpu/test_run_kernel_child.py -v
```

Expected: PASS, all tests in this file.

- [ ] **Step 7: Update `LocalGPURunner.benchmark_kernel` to consume `baseline_error` from child payload**

In `src/cuda_engine/services/gpu/local.py`, in the section that parses the child's output payload back into a `BenchmarkResult` (after `benchmark_kernel`'s subprocess call returns), add `baseline_error` to the constructed BenchmarkResult:

```python
return BenchmarkResult(
    ok=...,
    custom_ms=float(benchmark.get("custom_ms", 0.0)),
    baseline_ms=benchmark.get("baseline_ms"),
    baseline_error=benchmark.get("baseline_error"),         # NEW
    achieved_gbps=benchmark.get("achieved_gbps"),
    ...
)
```

(Grep `LocalGPURunner.benchmark_kernel` body in `local.py` to find the exact BenchmarkResult construction site; there's only one success path inside that method.)

- [ ] **Step 8: Run full suite**

```
python -m pytest --ignore=tests/integration
```

Expected: same passing count + the 2 new tests in this task.

- [ ] **Step 9: Lint + types**

```
python -m ruff check src tests
python -m mypy src
```

Expected: clean.

- [ ] **Step 10: Commit**

```bash
git add src/cuda_engine/services/gpu/_run_kernel_child.py src/cuda_engine/services/gpu/local.py tests/unit/services/gpu/test_run_kernel_child.py
git commit -m "feat(gpu): real torch.compile baseline in subprocess child"
```

---

## Chunk 3: `_speedup` returns None, PerformanceReport schema change, Stage 4 reference threading

This is the behaviorally biggest change. `speedup` becomes nullable downstream.

### Task 3.1: `_speedup` returns Optional[float]

**Files:**
- Modify: `src/cuda_engine/stages/performance.py` (the `_speedup` helper)
- Modify: `tests/unit/stages/test_performance.py` (extend)

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/stages/test_performance.py`:

```python
def test_speedup_returns_none_when_baseline_missing() -> None:
    from cuda_engine.stages.performance import _speedup

    assert _speedup(baseline_ms=None, custom_ms=0.01) is None


def test_speedup_returns_none_when_custom_ms_nonpositive() -> None:
    from cuda_engine.stages.performance import _speedup

    assert _speedup(baseline_ms=0.01, custom_ms=0.0) is None
    assert _speedup(baseline_ms=0.01, custom_ms=-1.0) is None


def test_speedup_returns_ratio_on_normal_inputs() -> None:
    from cuda_engine.stages.performance import _speedup

    assert _speedup(baseline_ms=2.0, custom_ms=1.0) == 2.0
```

- [ ] **Step 2: Run to confirm RED on `test_speedup_returns_none_when_baseline_missing`**

```
python -m pytest tests/unit/stages/test_performance.py::test_speedup_returns_none_when_baseline_missing -v
```

Expected: FAIL — current implementation returns `1.0`.

- [ ] **Step 3: Update `_speedup`**

In `src/cuda_engine/stages/performance.py`:

```python
def _speedup(*, baseline_ms: float | None, custom_ms: float) -> float | None:
    if custom_ms <= 0 or baseline_ms is None:
        return None
    return baseline_ms / custom_ms
```

- [ ] **Step 4: Run new tests to confirm GREEN**

```
python -m pytest tests/unit/stages/test_performance.py::test_speedup_returns_none_when_baseline_missing tests/unit/stages/test_performance.py::test_speedup_returns_none_when_custom_ms_nonpositive tests/unit/stages/test_performance.py::test_speedup_returns_ratio_on_normal_inputs -v
```

Expected: PASS.

- [ ] **Step 5: Don't commit yet — full suite will fail because callers of `_speedup` don't handle None.** Continue to Task 3.2.

### Task 3.2: Update `PerformanceReport` schema to nullable speedups

**Files:**
- Modify: `src/cuda_engine/models/reports.py`
- Modify: callers/serializers as needed

- [ ] **Step 1: Update the model**

In `src/cuda_engine/models/reports.py`:

```python
class PerformanceReport(BaseModel):
    speedup_vs_reference: float | None = None       # was: float
    speedup_vs_torch_compile: float | None = None   # was: float
    achieved_tflops: float | None = None
    achieved_gbps: float | None = None
    occupancy: float | None = None
    regs_per_thread: int | None = None
    spill_bytes: int = 0
    below_target: bool = False
    notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
```

- [ ] **Step 2: Update `Stage4Performance.run()` to handle Optional speedups**

In `src/cuda_engine/stages/performance.py`, several construction sites of `PerformanceReport`:

- Around line 40-46 (initial early-return when `kernel_so_path is None`): replace `speedup_vs_reference=0.0, speedup_vs_torch_compile=0.0` with `speedup_vs_reference=None, speedup_vs_torch_compile=None`.
- Around line 63-71 (benchmark failure): same replacement.
- Around line 126-133 (success path): use `current_speedup` directly (it's now Optional). `below_target` computation needs an update:

```python
below_target = current_speedup is None or current_speedup < target
report = PerformanceReport(
    speedup_vs_reference=current_speedup,
    speedup_vs_torch_compile=current_speedup,
    achieved_gbps=current_benchmark.achieved_gbps,
    below_target=below_target,
    warnings=warnings,
    notes=notes,
)
```

- [ ] **Step 3: Update the perf retry loop entry condition**

The entry gate (post-prior-chunk):

```python
if self.llm is not None and retry_budget > 0:
    ... self._retry_loop(...)
```

stays unchanged. The retry-loop body's internal comparisons need updating:

```python
# Inside _retry_loop, where best_speedup/current_speedup are compared
if new_speedup is not None and (best_speedup is None or new_speedup > best_speedup):
    best_artifact = candidate
    best_benchmark = new_benchmark
    best_speedup = new_speedup
```

And the terminal warning:

```python
if best_speedup is None or best_speedup < target:
    warnings.append(
        f"perf retry budget exhausted: best speedup {best_speedup if best_speedup is not None else 'unmeasured'} below target {target:.3f}"
    )
```

Update the per-attempt `notes` string to handle `None`:

```python
def _fmt(s: float | None) -> str:
    return f"{s:.3f}" if s is not None else "n/a"

next_best = max(filter(None, [best_speedup, new_speedup]), default=None)
notes.append(
    f"perf_repair attempt {attempt}: speedup {_fmt(current_speedup)} -> "
    f"{_fmt(new_speedup)} (best={_fmt(next_best)})"
)
```

`filter(None, [...])` drops `None` values; `max(default=None)` handles all-None case. (`filter(None, ...)` also drops `0.0` — but speedup of `0` is non-physical here since `_speedup` returns `None` for `custom_ms <= 0`.)

- [ ] **Step 4: Update CLI rendering**

Search for code that prints `speedup_vs_torch_compile`:

```
grep -rn "speedup_vs_torch_compile" src/cuda_engine/ | grep -v test_
```

Any printf/format site needs to handle `None`. The CLI in `src/cuda_engine/cli.py` likely formats this — render `None` as `"n/a"`.

- [ ] **Step 5: Run all unit tests to find broken assertions**

```
python -m pytest --ignore=tests/integration -v 2>&1 | grep -E "FAIL|ERROR" | head -30
```

Expected: several failures in `test_performance.py` (assertions like `report.speedup_vs_torch_compile == 2.0` still work; assertions in tests like `test_stage4_performance_reports_missing_shared_object` that expected `0.0` will fail and need updating to `None`).

- [ ] **Step 6: Fix existing test assertions to match new schema**

For each test that fails due to expecting `0.0`:
- Where the test simulates an unmeasured-baseline (e.g., kernel_so_path is None, benchmark failed): assertion becomes `report.speedup_vs_torch_compile is None`.
- Where the test simulates a successful measurement (e.g., the `_retries_until_target_met` test with `baseline_ms=2.0, custom_ms=1.0`): assertion stays `report.speedup_vs_torch_compile == 2.0` — that case is unchanged.

Also: existing tests provide `BenchmarkResult` fixtures with explicit `baseline_ms=2.0`. Those continue producing real speedups. The change only affects tests that produced `None` baseline.

- [ ] **Step 7: Run unit suite to confirm GREEN**

```
python -m pytest --ignore=tests/integration
```

Expected: all passing.

- [ ] **Step 8: Lint + types**

```
python -m ruff check src tests
python -m mypy src
```

Expected: clean. mypy may flag spots where `float | None` arithmetic was assumed — fix those at the source by adding `None` checks.

- [ ] **Step 9: Commit (this is a big commit covering schema + speedup helper + all call site fixes)**

```bash
git add src/cuda_engine/stages/performance.py src/cuda_engine/models/reports.py src/cuda_engine/cli.py tests/unit/stages/test_performance.py
git commit -m "$(cat <<'EOF'
feat(stage4): _speedup returns None on missing baseline, schema goes nullable

Replaces the silent 1.0 fallback in _speedup with explicit None. Updates
PerformanceReport.speedup_vs_torch_compile/_vs_reference to float | None.
below_target now True when speedup is None or < target. Retry loop's
best-so-far tracking handles None correctly (None never wins, real value
beats None). CLI renders None as "n/a".

This is the key change that makes the M3 fast_1 metric honest: kernels
whose baseline measurement fails no longer silently report "matches
parity".
EOF
)"
```

### Task 3.3: Thread `reference` through Stage 4 + cache across retry loop

**Files:**
- Modify: `src/cuda_engine/stages/performance.py`
- Modify: `src/cuda_engine/orchestrator.py`
- Modify: `tests/unit/test_orchestrator.py`, `tests/unit/stages/test_performance.py`

- [ ] **Step 1: Write failing test for orchestrator → Stage 4 reference threading**

Append to `tests/unit/test_orchestrator.py`:

```python
def test_orchestrator_threads_reference_to_stage4_benchmark() -> None:
    """The reference function passed to synthesize() must reach gpu.benchmark_kernel."""
    torch = __import__("torch")
    store = InMemoryStore()

    def _identity_ref(x):
        return x

    orchestrator = Orchestrator(
        llm=MockLLMClient(
            responses=[
                SPEC_JSON,
                LLMResponse(
                    text="```cuda\nextern code\n```",
                    model="mock",
                    tool_calls=[
                        {"name": "compile_kernel", "input": {"src": "extern code", "target_arch": "sm_80"}}
                    ],
                ),
                "```cuda\n// annotated\nextern code\n```",
            ]
        ),
        gpu=MockGPURunner(
            compile_results=[CompileResult(ok=True, so_path=Path("kernel.so"), log="ok")],
            run_results=[
                RunResult(ok=True, output_tensors=[torch.arange(size, dtype=torch.float32)])
                for size in SHAPE_SIZES
            ],
        ),
        store=store,
        cfg=SynthesisConfig(retry_budgets=RetryBudgets(performance=0)),
    )

    orchestrator.run(prompt="noop", reference=_identity_ref, target="sm_80")

    benchmark_calls = orchestrator.gpu.benchmark_calls
    assert benchmark_calls, "Stage 4 must call benchmark_kernel at least once"
    assert benchmark_calls[0]["reference"] is _identity_ref, (
        "first benchmark must receive the orchestrator's reference"
    )
```

- [ ] **Step 2: Run to confirm RED**

Expected: FAIL — Stage 4 doesn't accept or pass `reference` yet.

- [ ] **Step 3: Add `reference` param to `Stage4Performance.run`**

```python
def run(
    self,
    *,
    spec: KernelSpec,
    artifact: KernelArtifact,
    run_id: str,
    retry_budget: int = 3,
    reference: Callable[..., Any] | None = None,    # NEW
) -> tuple[PerformanceReport, KernelArtifact]:
```

Add `Callable` import.

In the body, pass to the first `benchmark_kernel`:

```python
benchmark = self.gpu.benchmark_kernel(
    artifact.kernel_so_path,
    inputs,
    reference=reference,                              # NEW
    warmup_iterations=self.cfg.benchmark_warmup_iterations,
    timed_iterations=self.cfg.benchmark_timed_iterations,
)
```

Capture the baseline for reuse:

```python
cached_baseline_ms = benchmark.baseline_ms
cached_baseline_error = benchmark.baseline_error
```

Pass `cached_baseline_ms` into `_retry_loop`:

```python
if self.llm is not None and retry_budget > 0:
    current_artifact, current_benchmark, current_speedup, warnings, notes = self._retry_loop(
        spec=spec, artifact=current_artifact, benchmark=current_benchmark,
        speedup=current_speedup, target=target, inputs=inputs, run_id=run_id,
        retry_budget=retry_budget,
        model=self.cfg.sonnet_model, attempt_offset=0,
        baseline_ms=cached_baseline_ms,               # NEW
    )
```

Same for the Opus escalation invocation (later in `run()`):

```python
... = self._retry_loop(..., baseline_ms=cached_baseline_ms)
```

If `cached_baseline_error is not None`, append to warnings before report construction:

```python
if cached_baseline_error is not None:
    warnings.append(cached_baseline_error)
```

- [ ] **Step 4: Add `baseline_ms` to `_retry_loop` signature**

```python
def _retry_loop(
    self,
    *,
    spec: KernelSpec,
    artifact: KernelArtifact,
    benchmark: BenchmarkResult,
    speedup: float | None,                            # type updated
    target: float,
    inputs: list[Any],
    run_id: str,
    retry_budget: int,
    model: str,
    attempt_offset: int = 0,
    baseline_ms: float | None,                        # NEW
) -> tuple[KernelArtifact, BenchmarkResult, float | None, list[str], list[str]]:
```

Inside the loop, each retry's `benchmark_kernel` call passes `reference=None` (we've cached the baseline):

```python
new_benchmark = self.gpu.benchmark_kernel(
    candidate_so,
    inputs,
    reference=None,                                   # NEW (cached above)
    warmup_iterations=self.cfg.benchmark_warmup_iterations,
    timed_iterations=self.cfg.benchmark_timed_iterations,
)
```

And `new_speedup` uses the cached baseline:

```python
new_speedup = _speedup(baseline_ms=baseline_ms, custom_ms=new_benchmark.custom_ms)
```

- [ ] **Step 5: Update `Orchestrator.run` to pass `reference` to Stage 4**

In `src/cuda_engine/orchestrator.py`, around the Stage 4 invocation:

```python
performance, artifact = _run_traced_stage(
    stage_traces, llm, "performance",
    lambda: Stage4Performance(llm=llm, gpu=self.gpu, store=self.store, cfg=self.cfg).run(
        spec=spec, artifact=artifact, run_id=run_id,
        retry_budget=self.cfg.retry_budgets.performance,
        reference=reference,                          # NEW
    ),
)
```

- [ ] **Step 6: Add Stage 4 unit test for baseline reuse**

Append to `tests/unit/stages/test_performance.py`:

```python
def test_stage4_retry_loop_reuses_cached_baseline() -> None:
    """The retry loop must not re-measure baseline on each attempt."""
    store = InMemoryStore()
    artifact = _initial_artifact_in_store(store, "run123", "// initial slow")
    initial = BenchmarkResult(
        ok=True, custom_ms=4.0, baseline_ms=2.0, warmup_iterations=10, timed_iterations=50
    )  # 0.5x — triggers retry loop
    faster = BenchmarkResult(
        ok=True, custom_ms=1.0, baseline_ms=None,  # retry passes reference=None, child returns None
        warmup_iterations=10, timed_iterations=50,
    )
    gpu = MockGPURunner(
        compile_results=[CompileResult(ok=True, so_path=Path("/tmp/v2.so"), log="ok")],
        benchmark_results=[initial, faster],
        profile_results=[NsightMetrics(occupancy=0.5, regs_per_thread=64)],
    )
    llm = MockLLMClient([_llm_compile_response("// faster kernel")])
    cfg = SynthesisConfig(perf_target_speedup_vs_torch_compile=1.0, opus_retry_budget_performance=0)
    stage = Stage4Performance(llm=llm, gpu=gpu, store=store, cfg=cfg)

    def _ref(x):
        return x

    report, _ = stage.run(
        spec=_spec(), artifact=artifact, run_id="run123",
        retry_budget=1, reference=_ref,
    )

    # First benchmark received the reference; retry benchmark did NOT
    assert gpu.benchmark_calls[0]["reference"] is _ref
    assert gpu.benchmark_calls[1]["reference"] is None
    # Final speedup uses cached baseline 2.0 against retry custom_ms 1.0 → 2.0
    assert report.speedup_vs_torch_compile == 2.0


def test_stage4_below_target_when_baseline_failed() -> None:
    """When baseline measurement fails, speedup is None and below_target is True."""
    store = InMemoryStore()
    artifact = _initial_artifact_in_store(store, "run123", "// kernel")
    initial = BenchmarkResult(
        ok=True, custom_ms=0.01, baseline_ms=None,
        baseline_error="torch.compile baseline failed: RuntimeError: traced graph too large",
        warmup_iterations=10, timed_iterations=50,
    )
    gpu = MockGPURunner(benchmark_results=[initial])
    cfg = SynthesisConfig(perf_target_speedup_vs_torch_compile=1.0, opus_retry_budget_performance=0)
    stage = Stage4Performance(llm=None, gpu=gpu, store=store, cfg=cfg)

    def _ref(x):
        return x

    report, _ = stage.run(
        spec=_spec(), artifact=artifact, run_id="run123",
        retry_budget=0, reference=_ref,
    )

    assert report.speedup_vs_torch_compile is None
    assert report.below_target is True
    assert any("torch.compile baseline failed" in w for w in report.warnings)
```

- [ ] **Step 7: Run new tests to confirm GREEN**

```
python -m pytest tests/unit/test_orchestrator.py::test_orchestrator_threads_reference_to_stage4_benchmark tests/unit/stages/test_performance.py::test_stage4_retry_loop_reuses_cached_baseline tests/unit/stages/test_performance.py::test_stage4_below_target_when_baseline_failed -v
```

Expected: all PASS.

- [ ] **Step 8: Run full suite to verify no regressions**

```
python -m pytest --ignore=tests/integration
```

Expected: all passing.

- [ ] **Step 9: Lint + types**

```
python -m ruff check src tests
python -m mypy src
```

Expected: clean.

- [ ] **Step 10: Commit**

```bash
git add src/cuda_engine/stages/performance.py src/cuda_engine/orchestrator.py tests/unit/test_orchestrator.py tests/unit/stages/test_performance.py
git commit -m "feat(orchestrator+stage4): thread reference into perf benchmark, cache baseline"
```

---

## Chunk 4: Eval runner baseline_status + summary

Update CSV columns, EvalRow, classification logic, and summary markdown to reflect the new "baseline can be measured or fail" distinction.

### Task 4.1: Add `baseline_status` to EvalRow + CSV

**Files:**
- Modify: `evals/runner.py`
- Modify: `tests/unit/test_internal_eval_suite.py` (or wherever runner tests live)

- [ ] **Step 1: Write failing test for new column**

In `tests/unit/test_internal_eval_suite.py` (or create a new `tests/unit/evals/test_runner.py` if that's the convention — check):

```python
def test_eval_row_baseline_status_ok_when_baseline_measured(tmp_path: Path) -> None:
    """A successful baseline produces baseline_status='ok' and a populated speedup."""
    # Set up a minimal eval run where the synth result reports speedup_vs_torch_compile=1.5
    # Assert the resulting CSV row's baseline_status == "ok"


def test_eval_row_baseline_status_failed_when_baseline_error(tmp_path: Path) -> None:
    """When PerformanceReport.warnings contains a baseline error, baseline_status='failed'."""
    # Set up an eval run where the synth result reports None speedup + baseline failure warning
    # Assert baseline_status == "failed", failure_kind == "baseline_failed"
```

Use existing test fixtures for `_run_kernel`. Mock `synthesize_fn` to return a controlled `SynthesisResult`.

- [ ] **Step 2: Run to confirm RED**

```
python -m pytest tests/unit/test_internal_eval_suite.py -v
```

Expected: FAIL on the new tests — `baseline_status` doesn't exist on EvalRow yet.

- [ ] **Step 3: Add the field to `EvalRow` and `CSV_COLUMNS`**

In `evals/runner.py`:

```python
CSV_COLUMNS = [
    "kernel",
    "passed",
    "run_id",
    "failed_stage",
    "failure_reason",
    "failure_kind",
    "speedup_vs_torch_compile",
    "speedup_vs_reference",
    "below_target",
    "baseline_status",              # NEW
    "artifacts_dir",
    "regression",
]


@dataclass(frozen=True)
class EvalRow:
    kernel: str
    passed: bool
    run_id: str
    failed_stage: int | None
    failure_reason: str
    speedup_vs_torch_compile: float | None
    speedup_vs_reference: float | None
    below_target: bool | None
    artifacts_dir: str
    failure_kind: str = ""
    baseline_status: str = ""        # NEW: "ok" | "failed" | "skipped" | ""
    regression: str = ""
```

- [ ] **Step 4: Compute `baseline_status` in `_run_kernel`**

In `_run_kernel`, where `EvalRow` is constructed:

```python
def _baseline_status(performance: PerformanceReport | None) -> str:
    if performance is None:
        return ""
    if performance.speedup_vs_torch_compile is not None:
        return "ok"
    if any("torch.compile baseline failed" in w for w in performance.warnings):
        return "failed"
    return "skipped"
```

Use it when constructing the EvalRow:

```python
return EvalRow(
    ...,
    baseline_status=_baseline_status(performance),
    ...
)
```

- [ ] **Step 5: Update `_classify_failure` to surface `baseline_failed`**

In `_classify_failure`, if the row indicates baseline failure (any "torch.compile baseline failed" in failure_reason or warnings), classify as `"baseline_failed"`. Look at the existing function — it currently classifies by `failed_stage`. Add a branch that checks warnings:

This is tricky because `_classify_failure` currently only receives `failed_stage` and `failure_reason` strings. We'd need to thread `baseline_status` into it. Simpler: handle baseline_failed via baseline_status directly in the row's `failure_kind` post-processing:

```python
# In _run_kernel after computing the row, override failure_kind if baseline failed
if row.baseline_status == "failed":
    row = EvalRow(**{**asdict(row), "failure_kind": "baseline_failed"})
```

(Make sure to import `asdict` if not already.)

- [ ] **Step 6: Update `_row_to_csv` and `_row_from_json` to round-trip the new field**

Grep both functions and add `baseline_status` to the dict mapping.

- [ ] **Step 7: Run new tests to confirm GREEN**

```
python -m pytest tests/unit/test_internal_eval_suite.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add evals/runner.py tests/unit/test_internal_eval_suite.py
git commit -m "feat(evals): add baseline_status column + baseline_failed failure_kind"
```

### Task 4.2: Update `summary.md` to split fast_1 from baseline_failed

**Files:**
- Modify: `evals/runner.py` (`_write_markdown`, `_m3_metrics`)
- Modify: `tests/unit/test_internal_eval_suite.py`

- [ ] **Step 1: Write failing test**

```python
def test_summary_markdown_includes_baseline_failed_count(tmp_path: Path) -> None:
    """summary.md must report baseline_failed count separately from fast_1."""
    # Build EvalSummary with: 2 ok-baseline kernels (one fast_1, one below), 1 baseline_failed
    # Render summary.md
    # Assert content includes "baseline_failed: 1/3" or similar
    # Assert fast_1 denominator is 3 and count is 1
```

- [ ] **Step 2: Run to confirm RED**

Expected: FAIL.

- [ ] **Step 3: Update `_m3_metrics` to count baseline_failed**

```python
def _m3_metrics(rows: list[EvalRow]) -> dict[str, Any]:
    total = len(rows)
    passed = sum(1 for r in rows if r.passed)
    baseline_failed = sum(1 for r in rows if r.baseline_status == "failed")
    fast_1 = sum(
        1
        for r in rows
        if r.speedup_vs_torch_compile is not None and r.speedup_vs_torch_compile > 1.0
    )
    below_target = sum(1 for r in rows if r.below_target is True and r.baseline_status == "ok")
    # ... rest of metrics ...
    return {
        ...,
        "baseline_failed": baseline_failed,
        "fast_1": fast_1,
        "below_target": below_target,
    }
```

- [ ] **Step 4: Update `_write_markdown` to include the new line**

In the M3 Metrics block:

```python
lines = [
    "# CUDA Engine Eval Summary",
    "",
    f"Pass rate: {metrics['passed']}/{metrics['total']}",
    "",
    "## M3 Metrics",
    "",
    f"- Pass rate: {metrics['passed']}/{metrics['total']} ({metrics['pass_rate_pct']:.1f}%)",
    f"- Median speedup vs torch.compile: {_format_metric_speedup(metrics['median_speedup'])}",
    f"- P25 speedup vs torch.compile: {_format_metric_speedup(metrics['p25_speedup'])}",
    f"- fast_1 kernels (>1.0x with measured baseline): {metrics['fast_1']}/{metrics['total']}",
    f"- baseline_failed (not counted in fast_1): {metrics['baseline_failed']}/{metrics['total']}",   # NEW
    f"- Below target (with measured baseline): {metrics['below_target']}/{metrics['total']}",
    ...
]
```

- [ ] **Step 5: Median/P25 speedup should only count rows with `baseline_status == "ok"`**

Look at the existing `_m3_metrics` median/P25 computation. Filter the speedup list:

```python
speedups = [r.speedup_vs_torch_compile for r in rows if r.speedup_vs_torch_compile is not None]
```

(Likely already does this since `speedup_vs_torch_compile` is now `Optional`. If not, add the filter.)

- [ ] **Step 6: Run new tests + existing summary tests to confirm GREEN**

```
python -m pytest tests/unit/test_internal_eval_suite.py -v
```

Expected: PASS.

- [ ] **Step 7: Run full suite**

```
python -m pytest --ignore=tests/integration
```

Expected: clean.

- [ ] **Step 8: Lint + types**

```
python -m ruff check src tests evals
python -m mypy src
```

Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add evals/runner.py tests/unit/test_internal_eval_suite.py
git commit -m "feat(evals): summary.md splits fast_1 from baseline_failed for honest reporting"
```

---

## Chunk 5: Integration test for real torch.compile baseline

One Colab-gated integration test that proves the wiring works against real GPU + torch.compile.

### Task 5.1: Write the integration test

**Files:**
- Create: `tests/integration/test_baseline_torch_compile.py`

- [ ] **Step 1: Create the test file**

```python
"""Integration test for torch.compile baseline measurement.

Gated on real CUDA + nvcc. Confirms that LocalGPURunner.benchmark_kernel
captures a real torch.compile baseline for a known reference function,
and that intentional failures surface as baseline_error.
"""

import shutil
from pathlib import Path

import pytest

from cuda_engine.config import SynthesisConfig
from cuda_engine.services.gpu.local import LocalGPURunner


def _relu_reference(x):
    """Top-level def so it pickles cleanly into the subprocess child."""
    import torch
    return torch.relu(x)


@pytest.mark.integration
def test_baseline_torch_compile_succeeds_against_real_gpu() -> None:
    pytest.importorskip("torch")
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc not available")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    runner = LocalGPURunner(SynthesisConfig(artifact_root=".test_artifacts/baseline"))
    # Build a minimal CUDA relu kernel so we have an so_path to benchmark
    src = _torch_extension_relu()
    compile_result = runner.compile(src, target_arch="sm_80")
    assert compile_result.ok, compile_result.log
    assert compile_result.so_path is not None

    x = torch.randn(1 << 14, device="cuda", dtype=torch.float32)
    result = runner.benchmark_kernel(
        compile_result.so_path,
        [x],
        reference=_relu_reference,
        warmup_iterations=3,
        timed_iterations=20,
    )

    assert result.ok, result.stderr
    assert result.custom_ms > 0
    assert result.baseline_ms is not None, f"baseline failed: {result.baseline_error}"
    assert result.baseline_error is None
    assert 1e-3 <= result.baseline_ms <= 100.0  # sane range


def _bad_reference(x):
    """A reference that raises in torch.compile to test failure surfacing."""
    raise RuntimeError("intentional baseline failure for test")


@pytest.mark.integration
def test_baseline_torch_compile_failure_surfaces_error() -> None:
    pytest.importorskip("torch")
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc not available")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    runner = LocalGPURunner(SynthesisConfig(artifact_root=".test_artifacts/baseline_fail"))
    compile_result = runner.compile(_torch_extension_relu(), target_arch="sm_80")
    assert compile_result.ok

    x = torch.randn(1 << 14, device="cuda", dtype=torch.float32)
    result = runner.benchmark_kernel(
        compile_result.so_path,
        [x],
        reference=_bad_reference,
        warmup_iterations=2,
        timed_iterations=5,
    )

    assert result.ok, "custom kernel must still benchmark even if baseline fails"
    assert result.custom_ms > 0
    assert result.baseline_ms is None
    assert result.baseline_error is not None
    assert "torch.compile baseline failed" in result.baseline_error
    assert "RuntimeError" in result.baseline_error
    assert "intentional baseline failure" in result.baseline_error


def _torch_extension_relu() -> str:
    return r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void relu_kernel(const float* x, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

torch::Tensor forward(torch::Tensor x) {
    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n
    );
    return out;
}

TORCH_LIBRARY(cuda_engine, m) { m.def("forward(Tensor x) -> Tensor"); }
TORCH_LIBRARY_IMPL(cuda_engine, CUDA, m) { m.impl("forward", &forward); }
'''
```

- [ ] **Step 2: Verify the test collects and skips cleanly on local Windows**

```
python -m pytest tests/integration/test_baseline_torch_compile.py --collect-only -q
python -m pytest tests/integration/test_baseline_torch_compile.py -v
```

Expected: 2 tests collected, both SKIPPED on local Windows (no CUDA).

- [ ] **Step 3: Lint**

```
python -m ruff check tests/integration/test_baseline_torch_compile.py
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_baseline_torch_compile.py
git commit -m "test(integration): real torch.compile baseline against CUDA"
```

---

## Chunk 6: Colab eval validation + evidence

Manual checkpoint. Same pattern as prior eval runs.

### Task 6.1: Push branch and run on Colab

- [ ] **Step 1: Push**

```bash
git push origin m3/perf-loop
```

- [ ] **Step 2: On Colab Pro + A100, with ANTHROPIC_API_KEY:**

```python
%cd /content
!rm -rf Cuda-Engine
!git clone --branch m3/perf-loop --depth 5 https://github.com/shivnarainms22/Cuda-Engine.git
%cd Cuda-Engine
!pip install -e . --quiet

import os
from google.colab import userdata
os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")

OUT_DIR = "/content/drive/MyDrive/cuda-engine-evals/2026-05-12-real-baseline"
!mkdir -p $OUT_DIR
!cuda-engine eval --suite internal --out $OUT_DIR 2>&1 | tee $OUT_DIR/run.log
```

Note the `$OUT_DIR` (bash) vs `{OUT_DIR}` (Python f-string) distinction — use `$` form in the shell command.

- [ ] **Step 3: Inspect summary**

```bash
!cat $OUT_DIR/summary.md
```

Look for the new "baseline_failed" line in M3 Metrics block.

- [ ] **Step 4: Interpret**

Expected outcomes:
- `baseline_failed = 0` ideally; if non-zero, those kernels' reference functions have an op `torch.compile` can't handle. Diagnose by inspecting per-kernel baseline_error in `kernels/<name>.json`.
- `fast_1 = K/30` is now the **honest** number. Likely much lower than the previous stub-baseline numbers (4/30) because torch.compile is genuinely competitive on simple ops.
- pass_rate should stay ≥28/30 (no regressions in correctness).

### Task 6.2: Append evidence + update memory

**Files:**
- Modify: `docs/milestones/M3-evidence.md`
- Modify: `C:\Users\Shivnarain\.claude\projects\D--Cuda-Engine\memory\project_cuda_engine.md`

- [ ] **Step 1: Append a `## Real torch.compile Baseline Checkpoint` section to `M3-evidence.md`**

Include:
- Run date, environment, branch, commit SHA tested.
- Durable output dir + zip path.
- Full `summary.md` content.
- Specific contrast vs prior fake-baseline run: previously 4/30 fast_1 against vector-add stub; now K/30 fast_1 against real torch.compile.
- Per-kernel `baseline_error` strings if any kernels show baseline_failed.
- Verdict for the M3 fast_1 ≥10/30 gate.
- Recommendation: close M3 at the honest fast_1 number, OR keep optimizing with this measurement infrastructure now in place.

- [ ] **Step 2: Commit evidence**

```bash
git add docs/milestones/M3-evidence.md
git commit -m "docs(m3): real torch.compile baseline eval evidence"
git push origin m3/perf-loop
```

- [ ] **Step 3: Update memory**

In `project_cuda_engine.md`, note:
- The fake-baseline bug discovery (key learning for future).
- The new honest fast_1 number.
- The M3 gate status (closed / pivot / continue).
- Next planned task.

---

## Final verification

After all 6 chunks:

- [ ] `python -m pytest --ignore=tests/integration` — expect ≥140 passed (132 baseline + ~8 new).
- [ ] `python -m ruff check src tests evals` — clean.
- [ ] `python -m mypy src` — clean.
- [ ] Integration test runs and passes on Colab A100 (Task 6.1).
- [ ] Evidence appended to `M3-evidence.md`.
- [ ] Memory updated.

---

## Open questions (resolve during execution)

- **`mode="reduce-overhead"` vs `mode="default"` for `torch.compile`.** Spec recommends reduce-overhead. If post-eval shows the baseline is unrealistically fast (CUDA graph capture wins by a lot on tiny kernels), switch to default mode for kernels below some shape threshold. Defer until data demands it.
- **Reference function pickling failure modes.** Eval kernels use top-level functions which pickle fine. If a future caller passes a closure, surface as `baseline_failed` with the pickle error — already handled by the try/except in `_measure_torch_compile_baseline` (it'll fire when the pickle in `local.py` raises). Actually: pickle happens in PARENT before subprocess spawn. So pickling failure would raise out of `benchmark_kernel`, not be captured as `baseline_error`. Either: (a) wrap the pickle dump in try/except and treat the failure as `baseline_error`, (b) document that public API requires pickleable references. Going with (b) is simpler — public API doc + clear error message. Update docstring on `benchmark_kernel` to note this constraint.
- **`baseline_ms` returned exactly `0.0`.** `_speedup` currently treats `custom_ms <= 0` as None but not `baseline_ms <= 0`. If a freak `torch.compile` produces 0ms, we'd get division by zero. Edge case unlikely on GPU — defer with a TODO if it surfaces.
