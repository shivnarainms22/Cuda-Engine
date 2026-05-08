# Stage Sonnet→Opus Escalation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire Sonnet→Opus escalation into Stage 2 codegen (catch `BudgetExhaustedError`, retry with Opus) and Stage 4 performance (after Sonnet retry loop ends below target, run additional Opus iterations of the same loop). Bundle a Colab integration test that exercises the Stage 4 escalation path end-to-end against real Anthropic + A100 + ncu.

**Architecture:** Two integration points by design — codegen escalation lives at orchestrator level (catches stage-boundary `BudgetExhaustedError`); Stage 4 escalation lives inside the stage (extends the existing `_retry_loop` with a different model). Pre-work parameterizes hardcoded model strings on `interview` / `codegen` / `polish` stages so the wrapper can swap models. New `SonnetFailureSummary` dataclass carries the last failed compile state from Sonnet into Opus's initial prompt.

**Tech Stack:** Python 3.11+, Pydantic v2 (config), pytest + `MockLLMClient` / `MockGPURunner` for unit tests, Anthropic API + nvcc/ncu/A100 for integration test (Colab manual run).

**Spec:** `docs/superpowers/specs/2026-05-07-stage-escalation-design.md`

**Skills referenced:**
- @superpowers:test-driven-development — every task: failing test first, then minimal code, then commit.
- @superpowers:verification-before-completion — checkpoints pass only with real command output.
- @superpowers:systematic-debugging — when a step fails, diagnose root cause; don't paper over.

**Branch:** `m3/perf-loop` (already exists, design committed at `d46f45d`).

---

## File Structure (locked)

### Files modified
| Path | Change |
|---|---|
| `src/cuda_engine/config.py` | +2 fields: `opus_retry_budget_codegen`, `opus_retry_budget_performance` |
| `src/cuda_engine/stages/base.py` | +`SonnetFailureSummary` dataclass; structured `BudgetExhaustedError` |
| `src/cuda_engine/stages/codegen.py` | `model: str` keyword-only param; `escalation_context` param; raise structured `BudgetExhaustedError` |
| `src/cuda_engine/stages/interview.py` | `model: str` keyword-only param replaces hardcode |
| `src/cuda_engine/stages/polish.py` | `model: str` keyword-only param replaces hardcode |
| `src/cuda_engine/stages/performance.py` | `_retry_loop` accepts `model` + `attempt_offset`; `run()` calls it twice on escalation |
| `src/cuda_engine/orchestrator.py` | new `_run_codegen_with_escalation` helper; routes both codegen call sites through it; passes explicit `model=cfg.sonnet_model` to interview/polish |
| `tests/unit/stages/test_codegen.py` | update existing tests to pass `model=`; add 3 new tests |
| `tests/unit/stages/test_performance.py` | add 2 new tests (model+offset, target-met-skips-escalation) |
| `tests/unit/test_orchestrator.py` | add 4 new tests (codegen escalation, escalation disabled, perf escalation, perf escalation disabled) |
| `tests/unit/test_config.py` | add new field assertions |
| `tests/unit/stages/test_interview.py` | update existing tests to pass `model=` |
| `tests/unit/stages/test_polish.py` | update existing tests to pass `model=` |

### Files created
| Path | Purpose |
|---|---|
| `tests/integration/test_e2e_perf_loop_escalation.py` | Real-Anthropic + A100 + ncu integration test for Stage 4 escalation |

---

## Chunk 1: Stage refactor — parameterize hardcoded model strings

Mechanical refactor that makes interview, codegen, and polish accept `model: str` as a keyword-only parameter instead of hardcoding `"claude-sonnet-4-6"`. Lands first so the escalation wrapper has a knob to twist. No behavioral change — orchestrator passes `cfg.sonnet_model` everywhere, output is byte-identical.

### Task 1.1: Make `Stage1Interview.run` accept `model` param

**Files:**
- Modify: `src/cuda_engine/stages/interview.py:17-52`
- Modify: `tests/unit/stages/test_interview.py` (update all callsites)

- [ ] **Step 1: Read current test file to understand existing call shape**

```bash
# We need to see how existing tests call .run() so we know exactly what to update
```

Read: `tests/unit/stages/test_interview.py`

- [ ] **Step 2: Update existing test calls to pass `model="claude-sonnet-4-6"` explicitly (all RED first)**

For each `stage.run(...)` call in `tests/unit/stages/test_interview.py`, add `model="claude-sonnet-4-6"` as a keyword argument. Save without running yet.

- [ ] **Step 3: Run test to confirm it fails on missing argument**

Run: `python -m pytest tests/unit/stages/test_interview.py -v`

Expected: FAIL with `TypeError: run() got an unexpected keyword argument 'model'` (Stage1Interview hasn't been updated yet).

- [ ] **Step 4: Modify `Stage1Interview.run` signature and body**

In `src/cuda_engine/stages/interview.py`:
- Add `model: str` as a required keyword-only param to `.run()` (after `target_arch`, before `run_id` — alphabetical-ish, but the existing convention puts `run_id` last).
- Replace `model="claude-sonnet-4-6"` on line 51 with `model=model`.

Resulting signature:
```python
def run(
    self,
    *,
    prompt: str,
    reference: Callable[..., Any],
    target_arch: str,
    run_id: str,
    model: str,
) -> KernelSpec:
```

- [ ] **Step 5: Run interview tests to confirm GREEN**

Run: `python -m pytest tests/unit/stages/test_interview.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cuda_engine/stages/interview.py tests/unit/stages/test_interview.py
git commit -m "refactor(stage1): accept model kwarg instead of hardcoding sonnet"
```

### Task 1.2: Make `Stage2Codegen.run` accept `model` param

**Files:**
- Modify: `src/cuda_engine/stages/codegen.py:15-23, 48-53`
- Modify: `tests/unit/stages/test_codegen.py:52,70,98,127`

- [ ] **Step 1: Update each `stage.run(...)` callsite in the existing test file**

In `tests/unit/stages/test_codegen.py`, every `stage.run(...)` invocation gains `model="claude-sonnet-4-6"`. Specifically lines 52, 70, 98, 127 — verify by reading the file.

- [ ] **Step 2: Run test to confirm RED**

Run: `python -m pytest tests/unit/stages/test_codegen.py -v`

Expected: FAIL with `TypeError: run() got an unexpected keyword argument 'model'`.

- [ ] **Step 3: Modify `Stage2Codegen.run` signature and body**

In `src/cuda_engine/stages/codegen.py`:
- Add `model: str` as a required keyword-only param to `.run()`.
- Replace `model="claude-sonnet-4-6"` on line 52 with `model=model`.

Resulting signature:
```python
def run(
    self,
    *,
    spec: KernelSpec,
    run_id: str,
    model: str,
    retry_budget: int = 3,
    repair_context: CorrectnessReport | None = None,
    artifact_prefix: str = "stage2_codegen",
) -> KernelArtifact:
```

- [ ] **Step 4: Run codegen tests to confirm GREEN**

Run: `python -m pytest tests/unit/stages/test_codegen.py -v`

Expected: PASS, all 4 existing tests.

- [ ] **Step 5: Commit**

```bash
git add src/cuda_engine/stages/codegen.py tests/unit/stages/test_codegen.py
git commit -m "refactor(stage2): accept model kwarg instead of hardcoding sonnet"
```

### Task 1.3: Make `Stage5Polish.run` accept `model` param

**Files:**
- Modify: `src/cuda_engine/stages/polish.py:13-23, 41-46`
- Modify: `tests/unit/stages/test_polish.py` (update all callsites)

- [ ] **Step 1: Update each `stage.run(...)` callsite in the polish test file**

In `tests/unit/stages/test_polish.py`, every `stage.run(...)` gains `model="claude-sonnet-4-6"`.

- [ ] **Step 2: Run test to confirm RED**

Run: `python -m pytest tests/unit/stages/test_polish.py -v`

Expected: FAIL with TypeError on missing `model` kwarg.

- [ ] **Step 3: Modify `Stage5Polish.run` signature and body**

In `src/cuda_engine/stages/polish.py`:
- Add `model: str` as a required keyword-only param.
- Replace `model="claude-sonnet-4-6"` on line 45 with `model=model`.

- [ ] **Step 4: Run polish tests to confirm GREEN**

Run: `python -m pytest tests/unit/stages/test_polish.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cuda_engine/stages/polish.py tests/unit/stages/test_polish.py
git commit -m "refactor(stage5): accept model kwarg instead of hardcoding sonnet"
```

### Task 1.4: Update orchestrator to pass `model=cfg.sonnet_model` to all three stages

**Files:**
- Modify: `src/cuda_engine/orchestrator.py:55-66, 100-106, 164-172`
- Verify: `tests/unit/test_orchestrator.py` (no updates needed — exercises through the orchestrator)

- [ ] **Step 1: Run orchestrator tests to confirm CURRENT BREAKAGE**

Run: `python -m pytest tests/unit/test_orchestrator.py -v`

Expected: FAIL — orchestrator calls `Stage1Interview/Stage2Codegen/Stage5Polish .run()` without `model` kwarg. (After tasks 1.1-1.3 the stages now require it.)

- [ ] **Step 2: Update orchestrator to pass `model=self.cfg.sonnet_model`**

In `src/cuda_engine/orchestrator.py`:
- `Stage1Interview(...).run(...)` (around line 55-60): add `model=self.cfg.sonnet_model`.
- `Stage2Codegen(...).run(...)` initial call (around line 66-70): add `model=self.cfg.sonnet_model`.
- `Stage2Codegen(...).run(...)` inside `repair_action` (around line 100-106): add `model=self.cfg.sonnet_model`.
- `Stage5Polish(...).run(...)` (around line 164-172): add `model=self.cfg.sonnet_model`.

- [ ] **Step 3: Run orchestrator tests to confirm GREEN**

Run: `python -m pytest tests/unit/test_orchestrator.py -v`

Expected: PASS — all existing orchestrator tests pass without behavior change.

- [ ] **Step 4: Run full unit suite as a sanity sweep**

Run: `python -m pytest -q --ignore=tests/integration`

Expected: 97 passed (matches pre-refactor count).

- [ ] **Step 5: Commit**

```bash
git add src/cuda_engine/orchestrator.py
git commit -m "refactor(orchestrator): pass cfg.sonnet_model to interview/codegen/polish"
```

---

## Chunk 2: Structured `BudgetExhaustedError` + `SonnetFailureSummary`

Augment the error raised by codegen with a structured payload so the orchestrator-level wrapper can pass it into Opus's initial prompt. Internal-only (never crosses public API), so a frozen `@dataclass`, not Pydantic.

### Task 2.1: Add `SonnetFailureSummary` dataclass and structured error

**Files:**
- Modify: `src/cuda_engine/stages/base.py`
- Modify: `tests/unit/stages/test_codegen.py` — extend `test_stage2_codegen_raises_when_retry_budget_exhausted`

- [ ] **Step 1: Write failing test for structured error**

Append to `tests/unit/stages/test_codegen.py`:

```python
def test_stage2_codegen_budget_exhausted_carries_summary() -> None:
    store = InMemoryStore()
    stage = Stage2Codegen(
        llm=MockLLMClient([_compile_call("bad1"), _compile_call("bad2")]),
        gpu=MockGPURunner(
            compile_results=[
                CompileResult(ok=False, log="log-1", errors=["err-1"]),
                CompileResult(ok=False, log="log-2", errors=["err-2"]),
            ]
        ),
        store=store,
    )

    with pytest.raises(BudgetExhaustedError) as exc_info:
        stage.run(spec=_spec(), run_id="run123", retry_budget=2, model="claude-sonnet-4-6")

    summary = exc_info.value.summary
    assert summary is not None
    assert summary.attempts_made == 2
    assert "err-2" in summary.last_compile_errors
    assert "log-2" in summary.last_compile_log
    assert summary.last_source_attempt == "bad2"
```

- [ ] **Step 2: Run test to confirm RED**

Run: `python -m pytest tests/unit/stages/test_codegen.py::test_stage2_codegen_budget_exhausted_carries_summary -v`

Expected: FAIL with `AttributeError: 'BudgetExhaustedError' object has no attribute 'summary'`.

- [ ] **Step 3: Add `SonnetFailureSummary` and structured error to `stages/base.py`**

Replace `src/cuda_engine/stages/base.py` content with:

```python
from dataclasses import dataclass

from cuda_engine.services.gpu.base import GPURunner
from cuda_engine.services.llm.base import LLMClient
from cuda_engine.services.store.base import ArtifactStore


@dataclass(frozen=True)
class SonnetFailureSummary:
    """Captures Sonnet's final failed-attempt state for handoff to Opus."""

    last_compile_errors: str
    last_compile_log: str
    last_source_attempt: str
    attempts_made: int


class BudgetExhaustedError(RuntimeError):
    """Raised when a stage exhausts its retry budget without producing a valid result."""

    def __init__(self, message: str, summary: SonnetFailureSummary | None = None) -> None:
        super().__init__(message)
        self.summary = summary


class StructuralStageError(RuntimeError):
    """Raised when a stage cannot produce structurally valid data."""


class Stage:
    name: str = "stage"

    def __init__(
        self,
        llm: LLMClient | None = None,
        gpu: GPURunner | None = None,
        store: ArtifactStore | None = None,
    ) -> None:
        self.llm = llm
        self.gpu = gpu
        self.store = store
```

- [ ] **Step 4: Update `Stage2Codegen` to populate the summary on bust**

In `src/cuda_engine/stages/codegen.py`:

Track the last source attempt across the loop. Replace the `raise BudgetExhaustedError(...)` block (currently lines 115-118) with:

```python
last_src = src if 'src' in dir() else ""  # final attempt's source
errors_str = "" if last_result is None else "\n".join(last_result.errors)
log_str = "" if last_result is None else last_result.log
summary = SonnetFailureSummary(
    last_compile_errors=errors_str,
    last_compile_log=log_str,
    last_source_attempt=last_src,
    attempts_made=retry_budget,
)
raise BudgetExhaustedError(
    f"codegen exhausted retry budget after {retry_budget} attempts: "
    f"{_exhausted_budget_detail(last_result)}",
    summary=summary,
)
```

Cleaner approach (preferred): pull `src` out of the loop scope by initializing `last_src = ""` before the for-loop and updating it after `_source_from_response`. Add the import:

```python
from cuda_engine.stages.base import BudgetExhaustedError, SonnetFailureSummary, Stage
```

- [ ] **Step 5: Run new test to confirm GREEN**

Run: `python -m pytest tests/unit/stages/test_codegen.py::test_stage2_codegen_budget_exhausted_carries_summary -v`

Expected: PASS.

- [ ] **Step 6: Run full codegen test file**

Run: `python -m pytest tests/unit/stages/test_codegen.py -v`

Expected: PASS, all 5 tests (4 existing + 1 new).

- [ ] **Step 7: Commit**

```bash
git add src/cuda_engine/stages/base.py src/cuda_engine/stages/codegen.py tests/unit/stages/test_codegen.py
git commit -m "feat(stages): SonnetFailureSummary on BudgetExhaustedError"
```

### Task 2.2: Make `Stage2Codegen` accept and use `escalation_context`

**Files:**
- Modify: `src/cuda_engine/stages/codegen.py`
- Modify: `tests/unit/stages/test_codegen.py`

- [ ] **Step 1: Write failing test for escalation_context prompt injection**

Append to `tests/unit/stages/test_codegen.py`:

```python
def test_stage2_codegen_escalation_context_appears_in_initial_prompt() -> None:
    from cuda_engine.stages.base import SonnetFailureSummary

    store = InMemoryStore()
    llm = MockLLMClient([_compile_call("ok")])
    stage = Stage2Codegen(
        llm=llm,
        gpu=MockGPURunner(
            compile_results=[CompileResult(ok=True, so_path=Path("/tmp/k.so"), log="ok")]
        ),
        store=store,
    )
    summary = SonnetFailureSummary(
        last_compile_errors="undefined symbol foo",
        last_compile_log="error log content",
        last_source_attempt="__global__ void bad() { foo(); }",
        attempts_made=3,
    )

    stage.run(
        spec=_spec(),
        run_id="run123",
        retry_budget=1,
        model="claude-opus-4-7",
        escalation_context=summary,
    )

    initial_prompt = llm.calls[0]["messages"][0]["content"]
    assert "previous attempts" in initial_prompt.lower() or "sonnet" in initial_prompt.lower()
    assert "undefined symbol foo" in initial_prompt
    assert "__global__ void bad() { foo(); }" in initial_prompt
    assert "3" in initial_prompt  # attempts_made
```

- [ ] **Step 2: Run to confirm RED**

Run: `python -m pytest tests/unit/stages/test_codegen.py::test_stage2_codegen_escalation_context_appears_in_initial_prompt -v`

Expected: FAIL — `escalation_context` is an unexpected kwarg.

- [ ] **Step 3: Add `escalation_context` param + prompt-injection helper**

In `src/cuda_engine/stages/codegen.py`:

Update `.run()` signature:
```python
def run(
    self,
    *,
    spec: KernelSpec,
    run_id: str,
    model: str,
    retry_budget: int = 3,
    repair_context: CorrectnessReport | None = None,
    escalation_context: SonnetFailureSummary | None = None,
    artifact_prefix: str = "stage2_codegen",
) -> KernelArtifact:
```

Update `_initial_user_prompt` to take `escalation_context`:
```python
def _initial_user_prompt(
    *,
    spec: KernelSpec,
    repair_context: CorrectnessReport | None,
    escalation_context: SonnetFailureSummary | None = None,
) -> str:
    base = (
        "Generate kernel.cu for this KernelSpec, then call compile_kernel.\n\n"
        f"{spec.model_dump_json(indent=2)}"
        if repair_context is None
        else (
            "Repair kernel.cu for this KernelSpec. The previous kernel compiled but failed "
            "correctness. Use the correctness report to fix the implementation, then call "
            "compile_kernel with the repaired CUDA source.\n\n"
            f"KernelSpec:\n{spec.model_dump_json(indent=2)}\n\n"
            f"Correctness report:\n{repair_context.model_dump_json(indent=2)}"
        )
    )
    if escalation_context is None:
        return base
    return (
        f"{_escalation_preamble(escalation_context)}\n\n{base}"
    )


def _escalation_preamble(summary: SonnetFailureSummary) -> str:
    return (
        f"Previous attempts with claude-sonnet-4-6 failed {summary.attempts_made} times. "
        "Address the underlying issue rather than repeating the prior approach.\n\n"
        f"Last compile errors:\n{summary.last_compile_errors}\n\n"
        f"Last compile log:\n{summary.last_compile_log}\n\n"
        f"Last source attempt:\n```cuda\n{summary.last_source_attempt}\n```"
    )
```

Update the existing call: `_initial_user_prompt(spec=spec, repair_context=repair_context)` → `_initial_user_prompt(spec=spec, repair_context=repair_context, escalation_context=escalation_context)`.

- [ ] **Step 4: Run new test to confirm GREEN**

Run: `python -m pytest tests/unit/stages/test_codegen.py::test_stage2_codegen_escalation_context_appears_in_initial_prompt -v`

Expected: PASS.

- [ ] **Step 5: Run full codegen test file**

Run: `python -m pytest tests/unit/stages/test_codegen.py -v`

Expected: PASS, all 6 tests.

- [ ] **Step 6: Commit**

```bash
git add src/cuda_engine/stages/codegen.py tests/unit/stages/test_codegen.py
git commit -m "feat(stage2): accept escalation_context, inject into initial prompt"
```

---

## Chunk 3: Config additions

Add the two new budget fields. Tiny but its own commit so config defaults are reviewable in isolation.

### Task 3.1: Add `opus_retry_budget_codegen` and `opus_retry_budget_performance`

**Files:**
- Modify: `src/cuda_engine/config.py`
- Modify: `tests/unit/test_config.py`

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_config.py`:

```python
def test_synthesis_config_has_opus_escalation_budgets() -> None:
    from cuda_engine.config import SynthesisConfig

    cfg = SynthesisConfig()
    assert cfg.opus_retry_budget_codegen == 3
    assert cfg.opus_retry_budget_performance == 1
    assert cfg.escalate_to_opus_on_bust is True


def test_synthesis_config_opus_budgets_overridable() -> None:
    from cuda_engine.config import SynthesisConfig

    cfg = SynthesisConfig(opus_retry_budget_codegen=2, opus_retry_budget_performance=0)
    assert cfg.opus_retry_budget_codegen == 2
    assert cfg.opus_retry_budget_performance == 0
```

- [ ] **Step 2: Run to confirm RED**

Run: `python -m pytest tests/unit/test_config.py -v`

Expected: FAIL — fields don't exist.

- [ ] **Step 3: Add fields to `SynthesisConfig`**

In `src/cuda_engine/config.py`, add two fields after the existing `opus_model` line:

```python
opus_retry_budget_codegen: int = 3
opus_retry_budget_performance: int = 1
```

- [ ] **Step 4: Run to confirm GREEN**

Run: `python -m pytest tests/unit/test_config.py -v`

Expected: PASS, both new tests + existing config tests.

- [ ] **Step 5: Commit**

```bash
git add src/cuda_engine/config.py tests/unit/test_config.py
git commit -m "feat(config): add opus_retry_budget_codegen / opus_retry_budget_performance"
```

---

## Chunk 4: Codegen escalation in orchestrator

Add `_run_codegen_with_escalation` helper and route both codegen call sites through it.

### Task 4.1: Implement orchestrator escalation helper + tests

**Files:**
- Modify: `src/cuda_engine/orchestrator.py`
- Modify: `tests/unit/test_orchestrator.py`

- [ ] **Step 1: Write failing test for codegen escalation**

Append to `tests/unit/test_orchestrator.py`:

```python
def test_orchestrator_escalates_codegen_to_opus_on_bust() -> None:
    """Sonnet busts 3x on codegen, Opus succeeds 1st try → run completes via Opus."""
    torch = __import__("torch")
    store = InMemoryStore()

    def _fail_compile_response() -> LLMResponse:
        return LLMResponse(
            text="```cuda\nbroken\n```",
            model="mock",
            tool_calls=[
                {"name": "compile_kernel", "input": {"src": "broken", "target_arch": "sm_80"}}
            ],
        )

    def _ok_compile_response() -> LLMResponse:
        return LLMResponse(
            text="```cuda\ngood\n```",
            model="mock",
            tool_calls=[
                {"name": "compile_kernel", "input": {"src": "good", "target_arch": "sm_80"}}
            ],
        )

    orchestrator = Orchestrator(
        llm=MockLLMClient(
            responses=[
                SPEC_JSON,                                    # interview (sonnet)
                _fail_compile_response(),                     # codegen sonnet attempt 1
                _fail_compile_response(),                     # codegen sonnet attempt 2
                _fail_compile_response(),                     # codegen sonnet attempt 3 (bust)
                _ok_compile_response(),                       # codegen opus attempt 1
                "```cuda\n// annotated\ngood\n```",           # polish
            ]
        ),
        gpu=MockGPURunner(
            compile_results=[
                CompileResult(ok=False, log="bad1", errors=["err1"]),
                CompileResult(ok=False, log="bad2", errors=["err2"]),
                CompileResult(ok=False, log="bad3", errors=["err3"]),
                CompileResult(ok=True, so_path=Path("/tmp/kernel.so"), log="ok"),
                CompileResult(ok=True, so_path=Path("/tmp/kernel.so"), log="ok"),  # polish recompile
            ],
            run_results=[
                RunResult(ok=True, output_tensors=[torch.arange(size, dtype=torch.float32)])
                for size in SHAPE_SIZES * 2  # both initial correctness + polish-correctness
            ],
        ),
        store=store,
        cfg=SynthesisConfig(
            retry_budgets=RetryBudgets(codegen=3),
            performance_shape_n=256,
            benchmark_warmup_iterations=2,
            benchmark_timed_iterations=3,
        ),
    )

    result = orchestrator.run(prompt="noop", reference=lambda x: x, target="sm_80")

    assert result.passed is True
    codegen_trace = next(t for t in result.report.stage_traces if t.stage_name == "codegen")
    assert "claude-sonnet-4-6" in codegen_trace.model_used
    assert "claude-opus-4-7" in codegen_trace.model_used
    assert codegen_trace.attempts == 4  # 3 sonnet + 1 opus
    # Opus retry artifact lands under escalated/
    escalated_files = [
        key for key in store._files if "stage2_codegen/escalated/attempt_01/" in key[1]
    ]
    assert escalated_files, f"expected escalated/ files, got: {list(store._files.keys())[:20]}"
```

- [ ] **Step 2: Run to confirm RED**

Run: `python -m pytest tests/unit/test_orchestrator.py::test_orchestrator_escalates_codegen_to_opus_on_bust -v`

Expected: FAIL with `BudgetExhaustedError` (not caught by orchestrator).

- [ ] **Step 3: Implement `_run_codegen_with_escalation` helper**

In `src/cuda_engine/orchestrator.py`, add at module level (after `_TracingLLMClient` class or near other helpers):

```python
def _run_codegen_with_escalation(
    *,
    llm: _TracingLLMClient,
    gpu: GPURunner,
    store: ArtifactStore,
    cfg: SynthesisConfig,
    run_args: dict[str, Any],
) -> KernelArtifact:
    """Run Stage2Codegen with Sonnet, escalating to Opus on BudgetExhaustedError."""
    try:
        return Stage2Codegen(llm=llm, gpu=gpu, store=store).run(
            **run_args, model=cfg.sonnet_model
        )
    except BudgetExhaustedError as bust:
        if not cfg.escalate_to_opus_on_bust or cfg.opus_retry_budget_codegen <= 0:
            raise
        opus_run_args = {
            **run_args,
            "retry_budget": cfg.opus_retry_budget_codegen,
            "artifact_prefix": f"{run_args.get('artifact_prefix', 'stage2_codegen')}/escalated",
            "escalation_context": bust.summary,
        }
        return Stage2Codegen(llm=llm, gpu=gpu, store=store).run(
            **opus_run_args, model=cfg.opus_model
        )
```

Add imports at top of orchestrator if missing: `from cuda_engine.stages.base import BudgetExhaustedError`.

Update both codegen call sites in `Orchestrator.run`:

**Initial codegen** (around line 62-71): replace
```python
artifact = _run_traced_stage(
    stage_traces, llm, "codegen",
    lambda: Stage2Codegen(llm=llm, gpu=self.gpu, store=self.store).run(
        spec=spec, run_id=run_id, retry_budget=self.cfg.retry_budgets.codegen,
        model=self.cfg.sonnet_model,
    ),
)
```
with
```python
artifact = _run_traced_stage(
    stage_traces, llm, "codegen",
    lambda: _run_codegen_with_escalation(
        llm=llm, gpu=self.gpu, store=self.store, cfg=self.cfg,
        run_args={
            "spec": spec,
            "run_id": run_id,
            "retry_budget": self.cfg.retry_budgets.codegen,
        },
    ),
)
```

**Repair-loop codegen** (around line 95-106): replace
```python
def repair_action(...):
    return Stage2Codegen(...).run(
        spec=spec, run_id=run_id, retry_budget=self.cfg.retry_budgets.codegen,
        repair_context=correctness_report,
        artifact_prefix=f"{repair_prefix}/codegen",
        model=self.cfg.sonnet_model,
    )
```
with
```python
def repair_action(
    correctness_report: CorrectnessReport = correctness,
    repair_prefix: str = repair_dir,
) -> KernelArtifact:
    return _run_codegen_with_escalation(
        llm=llm, gpu=self.gpu, store=self.store, cfg=self.cfg,
        run_args={
            "spec": spec,
            "run_id": run_id,
            "retry_budget": self.cfg.retry_budgets.codegen,
            "repair_context": correctness_report,
            "artifact_prefix": f"{repair_prefix}/codegen",
        },
    )
```

- [ ] **Step 4: Run new test to confirm GREEN**

Run: `python -m pytest tests/unit/test_orchestrator.py::test_orchestrator_escalates_codegen_to_opus_on_bust -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cuda_engine/orchestrator.py tests/unit/test_orchestrator.py
git commit -m "feat(orchestrator): codegen Sonnet→Opus escalation wrapper"
```

### Task 4.2: Test "escalation disabled" path

**Files:**
- Modify: `tests/unit/test_orchestrator.py`

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_orchestrator.py`:

```python
def test_orchestrator_codegen_escalation_disabled_surfaces_bust() -> None:
    """With escalate_to_opus_on_bust=False, Sonnet bust propagates as Stage-3 failure."""
    torch = __import__("torch")
    store = InMemoryStore()

    def _fail_compile_response() -> LLMResponse:
        return LLMResponse(
            text="```cuda\nbroken\n```",
            model="mock",
            tool_calls=[
                {"name": "compile_kernel", "input": {"src": "broken", "target_arch": "sm_80"}}
            ],
        )

    orchestrator = Orchestrator(
        llm=MockLLMClient(
            responses=[
                SPEC_JSON,
                _fail_compile_response(), _fail_compile_response(), _fail_compile_response(),
            ]
        ),
        gpu=MockGPURunner(
            compile_results=[
                CompileResult(ok=False, log="bad1", errors=["err1"]),
                CompileResult(ok=False, log="bad2", errors=["err2"]),
                CompileResult(ok=False, log="bad3", errors=["err3"]),
            ],
        ),
        store=store,
        cfg=SynthesisConfig(
            escalate_to_opus_on_bust=False,
            retry_budgets=RetryBudgets(codegen=3),
        ),
    )

    import pytest
    from cuda_engine.stages.base import BudgetExhaustedError
    with pytest.raises(BudgetExhaustedError):
        orchestrator.run(prompt="noop", reference=lambda x: x, target="sm_80")
```

- [ ] **Step 2: Run to confirm GREEN immediately**

Run: `python -m pytest tests/unit/test_orchestrator.py::test_orchestrator_codegen_escalation_disabled_surfaces_bust -v`

Expected: PASS — the helper already checks `cfg.escalate_to_opus_on_bust`.

(If it fails, the helper logic needs the early `if not cfg.escalate_to_opus_on_bust: raise`. Re-read Task 4.1 Step 3.)

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_orchestrator.py
git commit -m "test(orchestrator): codegen escalation respects escalate_to_opus_on_bust=False"
```

---

## Chunk 5: Stage 4 perf-loop escalation

Extend `_retry_loop` to take `model` + `attempt_offset`, then call it twice from `run()` when below_target.

### Task 5.1: Parameterize `_retry_loop` on model and offset

**Files:**
- Modify: `src/cuda_engine/stages/performance.py:105-116, 134-244`
- Modify: `tests/unit/stages/test_performance.py`

- [ ] **Step 1: Write failing test for `_retry_loop` model + offset**

Append to `tests/unit/stages/test_performance.py`:

```python
def test_perf_retry_loop_uses_model_and_offset(tmp_path: Path) -> None:
    """_retry_loop should send the given model to LLM and number attempts after the offset."""
    # Use existing helpers from this file to construct a Stage4Performance + run _retry_loop
    # directly with a one-iteration budget. Verify (a) llm.calls[0]['model'] == 'claude-opus-4-7',
    # (b) artifact dir written under attempt_04/ when offset=3.
    # ... (test author: model after existing test_perf_retry_loop_skips_when_target_met or similar)
```

The full test body is filled out using existing helpers in `tests/unit/stages/test_performance.py` (look for `_make_stage()` or equivalent helper that constructs Stage4 with mocks). Pattern: prepare a single-iteration `_retry_loop` call with `attempt_offset=3` and `model="claude-opus-4-7"`. After call, assert:
- `llm.calls[-1]["model"] == "claude-opus-4-7"`
- The artifact written is under `stage4_performance/perf_repair/attempt_04/`

- [ ] **Step 2: Run to confirm RED**

Run: `python -m pytest tests/unit/stages/test_performance.py::test_perf_retry_loop_uses_model_and_offset -v`

Expected: FAIL — `_retry_loop` doesn't accept `model` or `attempt_offset` yet.

- [ ] **Step 3: Modify `_retry_loop` signature and body**

In `src/cuda_engine/stages/performance.py`:

Update `_retry_loop` signature:
```python
def _retry_loop(
    self,
    *,
    spec: KernelSpec,
    artifact: KernelArtifact,
    benchmark: BenchmarkResult,
    speedup: float,
    target: float,
    inputs: list[Any],
    run_id: str,
    retry_budget: int,
    model: str,
    attempt_offset: int = 0,
) -> tuple[KernelArtifact, BenchmarkResult, float, list[str], list[str]]:
```

Inside the body:
- Replace `attempt in range(1, retry_budget + 1)` with `for local_attempt in range(1, retry_budget + 1):` and `attempt = local_attempt + attempt_offset`.
- Replace `model=self.cfg.sonnet_model` (line 181) with `model=model`.
- All `f"…attempt_{attempt:02d}…"` strings now use the offset-adjusted `attempt` variable, so artifacts land at `attempt_(offset+1)..attempt_(offset+retry_budget)/`.

- [ ] **Step 4: Update existing `_retry_loop` call inside `Stage4Performance.run()`**

In the same file around line 82-92, the current call:
```python
current_artifact, current_benchmark, current_speedup, warnings, notes = self._retry_loop(
    spec=spec, artifact=current_artifact, benchmark=current_benchmark,
    speedup=current_speedup, target=target, inputs=inputs, run_id=run_id,
    retry_budget=retry_budget,
)
```

becomes:
```python
current_artifact, current_benchmark, current_speedup, warnings, notes = self._retry_loop(
    spec=spec, artifact=current_artifact, benchmark=current_benchmark,
    speedup=current_speedup, target=target, inputs=inputs, run_id=run_id,
    retry_budget=retry_budget,
    model=self.cfg.sonnet_model,
    attempt_offset=0,
)
```

- [ ] **Step 5: Run new test to confirm GREEN**

Run: `python -m pytest tests/unit/stages/test_performance.py::test_perf_retry_loop_uses_model_and_offset -v`

Expected: PASS.

- [ ] **Step 6: Run full performance test file to confirm no regressions**

Run: `python -m pytest tests/unit/stages/test_performance.py -v`

Expected: PASS, all existing perf tests + new one.

- [ ] **Step 7: Commit**

```bash
git add src/cuda_engine/stages/performance.py tests/unit/stages/test_performance.py
git commit -m "refactor(stage4): _retry_loop accepts model and attempt_offset"
```

### Task 5.2: Add Sonnet→Opus escalation in `Stage4Performance.run()`

**Files:**
- Modify: `src/cuda_engine/stages/performance.py:82-103`
- Modify: `tests/unit/stages/test_performance.py`
- Modify: `tests/unit/test_orchestrator.py`

- [ ] **Step 1: Write failing test for in-stage escalation (unit-level)**

Append to `tests/unit/stages/test_performance.py`:

```python
def test_stage4_escalates_to_opus_when_below_target() -> None:
    """When Sonnet loop ends below target, Stage 4 runs additional Opus iterations."""
    # Construct Stage4Performance with:
    #   - retry_budget=1, opus_retry_budget_performance=1
    #   - Sonnet attempt 1: speedup 0.5 (below target 1.0)
    #   - Opus attempt 1: speedup 1.5 (above target)
    # Assert:
    #   - 2 LLM calls total
    #   - llm.calls[0]['model'] == 'claude-sonnet-4-6'
    #   - llm.calls[1]['model'] == 'claude-opus-4-7'
    #   - report.below_target is False
    #   - 'escalated to opus' in any report.notes entry
    #   - attempt directories: attempt_01/ (sonnet) and attempt_02/ (opus)
```

Test body uses the existing perf-test fixtures — search the file for `MockLLMClient(...)` setups and adapt with two different benchmark results.

- [ ] **Step 2: Run to confirm RED**

Run: `python -m pytest tests/unit/stages/test_performance.py::test_stage4_escalates_to_opus_when_below_target -v`

Expected: FAIL — Stage4 only runs Sonnet; below_target=True with no escalation.

- [ ] **Step 3: Implement escalation in `Stage4Performance.run()`**

After the existing `_retry_loop` call (around line 92), insert before the `report = PerformanceReport(...)` construction:

```python
if (
    current_speedup < target
    and self.cfg.escalate_to_opus_on_bust
    and self.cfg.opus_retry_budget_performance > 0
    and self.llm is not None
):
    notes.append(
        f"escalated to opus after sonnet retry budget exhausted at speedup {current_speedup:.3f}"
    )
    (
        current_artifact,
        current_benchmark,
        current_speedup,
        opus_warnings,
        opus_notes,
    ) = self._retry_loop(
        spec=spec,
        artifact=current_artifact,
        benchmark=current_benchmark,
        speedup=current_speedup,
        target=target,
        inputs=inputs,
        run_id=run_id,
        retry_budget=self.cfg.opus_retry_budget_performance,
        model=self.cfg.opus_model,
        attempt_offset=retry_budget,
    )
    warnings.extend(opus_warnings)
    notes.extend(opus_notes)
```

- [ ] **Step 4: Run unit test to confirm GREEN**

Run: `python -m pytest tests/unit/stages/test_performance.py::test_stage4_escalates_to_opus_when_below_target -v`

Expected: PASS.

- [ ] **Step 5: Add `test_perf_escalation_skipped_when_disabled` to perf tests**

Append to `tests/unit/stages/test_performance.py`:

```python
def test_stage4_skips_escalation_when_disabled() -> None:
    """opus_retry_budget_performance=0 → no Opus loop, even when below_target."""
    # Same setup as the previous test BUT cfg.opus_retry_budget_performance=0.
    # Assert llm.call_count == 1 (sonnet only), report.below_target is True.
```

- [ ] **Step 6: Run to confirm GREEN**

Run: `python -m pytest tests/unit/stages/test_performance.py::test_stage4_skips_escalation_when_disabled -v`

Expected: PASS — the `opus_retry_budget_performance > 0` gate handles this.

- [ ] **Step 7: Add orchestrator-level test for perf escalation**

Append to `tests/unit/test_orchestrator.py`:

```python
def test_orchestrator_perf_stage_escalates_to_opus() -> None:
    """End-to-end through Orchestrator: perf below target on Sonnet → Opus iteration runs."""
    # Single-attempt perf_budget, single-attempt opus_perf_budget.
    # MockLLMClient: SPEC_JSON, codegen response, sonnet perf-fix response, opus perf-fix response, polish response
    # MockGPURunner with controlled benchmarks: initial below target, sonnet retry below target,
    #     opus retry above target.
    # Assert: result.passed True, perf trace model_used contains both sonnet+opus.
```

Test body modeled on the existing happy-path orchestrator test, with `cfg.retry_budgets.performance=1`, `cfg.opus_retry_budget_performance=1`, and benchmark fixtures that force the escalation path.

- [ ] **Step 8: Run new orchestrator test**

Run: `python -m pytest tests/unit/test_orchestrator.py::test_orchestrator_perf_stage_escalates_to_opus -v`

Expected: PASS.

- [ ] **Step 9: Run the full unit suite to confirm no regressions**

Run: `python -m pytest -q --ignore=tests/integration`

Expected: ~104 passed (97 baseline + ~7 new).

- [ ] **Step 10: Lint and type checks**

Run: `ruff check src tests`
Expected: no errors.

Run: `mypy src`
Expected: no errors.

- [ ] **Step 11: Commit**

```bash
git add src/cuda_engine/stages/performance.py tests/unit/stages/test_performance.py tests/unit/test_orchestrator.py
git commit -m "feat(stage4): Sonnet→Opus escalation when perf loop ends below target"
```

---

## Chunk 6: Colab integration test for perf escalation

End-to-end real-Anthropic + A100 + ncu test. Skipped on local CI; run manually on Colab Pro before declaring 4.3 done.

### Task 6.1: Write the integration test

**Files:**
- Create: `tests/integration/test_e2e_perf_loop_escalation.py`

- [ ] **Step 1: Read an existing integration test for the file shape**

Read: `tests/integration/test_e2e_rms_norm.py` (or `test_e2e_vector_add.py`) — pattern for `@pytest.mark.integration`, skip guards, and the `synthesize()` invocation.

- [ ] **Step 2: Create the new integration test**

Create `tests/integration/test_e2e_perf_loop_escalation.py`:

```python
import os
import shutil
from pathlib import Path

import pytest

from cuda_engine import SynthesisConfig, synthesize
from cuda_engine.config import RetryBudgets


def _slow_softmax_reference(x):
    """Reference for a deliberately optimization-rich kernel.

    Implementation note: softmax over last dim. The naïve CUDA implementation
    (per-row two-pass, no shared memory tiling) is well below 1.0× torch.compile
    on A100, exercising the perf retry loop. Opus then either tiles or uses
    warp-level reductions to cross the bar.
    """
    import torch

    return torch.softmax(x, dim=-1)


@pytest.mark.integration
def test_perf_loop_escalates_to_opus_on_softmax_e2e(tmp_path: Path) -> None:
    """End-to-end: run synthesize() on a kernel whose naive impl is slow.

    Assertions deliberately soft: don't require Opus to actually beat the target,
    only that the escalation path RAN. Whether Opus converges on a ≥1.0× kernel
    is a real-world variable not pinnable in CI.
    """
    pytest.importorskip("torch")
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc not available")
    if shutil.which("ncu") is None:
        pytest.skip("ncu not available")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cfg = SynthesisConfig(
        artifact_root=str(tmp_path),
        retry_budgets=RetryBudgets(codegen=2, correctness=2, performance=1),
        opus_retry_budget_performance=1,
        opus_retry_budget_codegen=1,
        performance_shape_n=4096,
        benchmark_warmup_iterations=3,
        benchmark_timed_iterations=10,
    )

    result = synthesize(
        prompt="Compute softmax over the last dim of a 2D float32 tensor.",
        reference=_slow_softmax_reference,
        target="sm_80",
        config=cfg,
    )

    # Run completed, even if perf bar wasn't beaten
    assert result.run_id

    perf_trace = next(
        (t for t in result.report.stage_traces if t.stage_name == "performance"),
        None,
    )
    assert perf_trace is not None, "performance stage missing from trace"

    # Soft assertion 1: escalation actually ran (both models touched the perf stage)
    if "claude-opus-4-7" not in perf_trace.model_used:
        pytest.skip(
            f"escalation did not trigger (Sonnet hit target on first try): "
            f"model_used={perf_trace.model_used}"
        )

    assert "claude-sonnet-4-6" in perf_trace.model_used
    assert "claude-opus-4-7" in perf_trace.model_used

    # Soft assertion 2: attempt directories exist for both Sonnet and Opus
    run_dir = Path(result.artifacts_dir)
    sonnet_attempt = run_dir / "stage4_performance" / "perf_repair" / "attempt_01"
    opus_attempt = run_dir / "stage4_performance" / "perf_repair" / "attempt_02"
    assert sonnet_attempt.exists(), f"missing {sonnet_attempt}"
    assert opus_attempt.exists(), f"missing {opus_attempt}"
    assert (opus_attempt / "nsight.json").exists()
    assert (opus_attempt / "benchmark.json").exists()

    # Soft assertion 3: notes record the escalation
    notes_str = " ".join(result.performance.notes) if result.performance else ""
    assert "escalated to opus" in notes_str.lower()
```

- [ ] **Step 3: Verify the test is collected and properly marked**

Run: `python -m pytest tests/integration/test_e2e_perf_loop_escalation.py --collect-only -q`

Expected: 1 test collected, marked `integration`.

- [ ] **Step 4: Run the test in the local sandbox to confirm it skips cleanly**

Run: `python -m pytest tests/integration/test_e2e_perf_loop_escalation.py -v`

Expected: SKIPPED (one or more of nvcc/ncu/ANTHROPIC_API_KEY/CUDA absent on local Windows). The test should NOT error during skip-decision.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_e2e_perf_loop_escalation.py
git commit -m "test(integration): Colab perf-loop Sonnet→Opus escalation end-to-end"
```

### Task 6.2: Run the integration test on Colab (manual checkpoint)

This task is not automatable — it requires the user to run on Colab Pro + A100.

- [ ] **Step 1: Push the branch**

Run: `git push origin m3/perf-loop`

- [ ] **Step 2: On Colab Pro (A100):**

```python
!git clone -b m3/perf-loop <repo-url>
%cd <repo>
!pip install -e .
!ANTHROPIC_API_KEY=sk-... pytest tests/integration/test_e2e_perf_loop_escalation.py -v -s
```

Expected outcome: PASS (escalation triggered, both attempt dirs exist, notes contain "escalated to opus") OR SKIP (Sonnet beat target on first try — acceptable variance for v1 perf bar).

- [ ] **Step 3: If the test fails**

Diagnose via `@superpowers:systematic-debugging`. Common scenarios:
- Sonnet's first kernel doesn't even compile → fix the prompt or the test kernel choice (use a softmax that's deliberately a known-low-perf shape).
- Opus runs but writes to a different attempt dir number → `attempt_offset` math is off; check `_retry_loop` offset logic.
- LLM call count differs from expectation → trace `model_used` field; re-check `_run_codegen_with_escalation` paths.

- [ ] **Step 4: Update plan with Colab evidence**

If Colab pass: append a `## Verification log` section to this plan with the date, Colab session URL/snippet, the actual `model_used` strings, and the speedups for both attempts.

If Colab failure that surfaces a code defect: file a follow-up task in this plan, fix on a branch off `m3/perf-loop`, re-run.

---

## Final verification

After all chunks complete:

- [ ] **Step 1: Full unit test suite green**

Run: `python -m pytest -q --ignore=tests/integration`
Expected: ~104 passed.

- [ ] **Step 2: Lint clean**

Run: `ruff check src tests`
Expected: no errors.

- [ ] **Step 3: Types clean**

Run: `mypy src`
Expected: no errors.

- [ ] **Step 4: Integration test confirmed on Colab (Task 6.2)**

Evidence appended to this plan or stored under `docs/milestones/M3-evidence.md`.

- [ ] **Step 5: Commit count check**

Run: `git log --oneline 4619978..HEAD`

Expected: ~10 commits (4 refactor, 2 feat-stage, 1 feat-config, 2 feat-orchestrator/stage4, 2 test).

If significantly more or fewer, re-read the commit-by-commit plan in design doc Section 9 and reconcile.

- [ ] **Step 6: Update memory**

Update `C:\Users\Shivnarain\.claude\projects\D--Cuda-Engine\memory\project_cuda_engine.md` with: Task 4.3 complete, escalation wired, Colab evidence link, next task = 4.5 (eval runner) per session recap.

---

## Open questions (resolve during execution)

- **Polish-stage model on escalated codegen runs.** When codegen escalates to Opus mid-run, Polish still runs on Sonnet (via `cfg.sonnet_model` in orchestrator). Acceptable for v1 — Polish is annotation-only, not optimization. If a future eval shows Polish corrupting Opus-produced kernels, revisit by passing the codegen's actual `model_used` to Polish.
- **Cache-control headers across model swaps.** Sonnet and Opus both honor Anthropic prompt-caching, but cache breakpoints are model-scoped. Switching models means cache miss on the system prompt. Expected ~$0.10-0.30 extra per escalated run; acceptable.
