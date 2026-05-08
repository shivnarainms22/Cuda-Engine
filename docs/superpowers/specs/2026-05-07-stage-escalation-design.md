# Stage Sonnet→Opus Escalation — Design Document

| Field | Value |
|---|---|
| **Status** | Approved (2026-05-07) |
| **Owner** | Shivnarain |
| **Branch** | `m3/perf-loop` |
| **Parent design** | `docs/superpowers/specs/2026-04-26-cuda-synthesis-engine-design.md` |
| **Implementation plan** | `docs/superpowers/plans/2026-05-07-stage-escalation-plan.md` (to be created) |
| **Milestone** | M3 — Task 4.3 |

---

## 1. Goal

When a stage exhausts its retry budget on Sonnet 4.6, automatically re-run the same work on Opus 4.7 before declaring failure. The intent is to push more eval kernels past the M3 quality bars (Stage 3 hard gate, Stage 4 ≥1.0× torch.compile soft gate) without bloating happy-path cost — Opus only runs when Sonnet has demonstrably failed.

Bundled in the same change: a Colab integration test that exercises the Stage 4 perf-loop escalation path end-to-end against real Anthropic + real A100 + real ncu, so we have one real-world verification on top of the unit tests.

---

## 2. Scope

### In scope
- **Stage 2 codegen escalation** on `BudgetExhaustedError`. Covers both the initial codegen call and every codegen invocation from the Stage 3 correctness-repair loop.
- **Stage 4 perf escalation** when the Sonnet retry loop ends with `current_speedup < target`. Triggers additional Opus iterations of the existing `_retry_loop`.
- Refactor of three stages (`codegen`, `interview`, `polish`) to read the model from config instead of hardcoding `"claude-sonnet-4-6"`.
- Unit tests for both escalation paths via `MockLLMClient`.
- One Colab integration test for the Stage 4 escalation path.

### Out of scope
- Stage 1 interview escalation. It's single-shot JSON generation; failures here are `StructuralStageError`, not budget exhaustion. YAGNI.
- Stage 5 polish escalation. Polish is non-blocking — failure does not affect product quality.
- Codegen-escalation real-Anthropic integration test. Forcing Sonnet to bust 3× consecutively in CI is too flaky; mocked unit tests are the right level for that logic.
- Multi-step escalation (Opus → some-other-model). Two-tier only.

### Explicitly rejected
- **Uniform orchestrator wrapper for all stages** (Approach B from brainstorm). Would require changing Stage 4's contract from soft-fail-with-warnings to raising `BudgetExhaustedError`, which forces re-translation logic in the wrapper to keep Polish running. Stage-internal escalation for Stage 4 reuses the existing perf-repair loop with zero contract change.
- **Carrying full Sonnet conversation history into Opus** (Approach C). Tool-use IDs and cache breakpoints don't translate cleanly across models. Structured failure summary in the user prompt is the right granularity.

---

## 3. Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │  SynthesisConfig                             │
                    │    escalate_to_opus_on_bust: bool = True     │
                    │    sonnet_model: str = "claude-sonnet-4-6"   │
                    │    opus_model:   str = "claude-opus-4-7"     │
                    │    opus_retry_budget_codegen: int = 3        │
                    │    opus_retry_budget_performance: int = 1    │
                    └──────────────────────────────────────────────┘
                                       │
                ┌──────────────────────┴────────────────────────┐
                │                                               │
   ┌────────────▼────────────┐                   ┌──────────────▼──────────────┐
   │  Orchestrator           │                   │  Stage4Performance.run()    │
   │  _run_codegen_with_     │                   │                             │
   │     escalation()        │                   │  loop1 = _retry_loop(       │
   │                         │                   │      model=sonnet, …)       │
   │  try Sonnet             │                   │                             │
   │   except BudgetExhausted│                   │  if loop1.below_target      │
   │      → run with Opus    │                   │     and escalate:           │
   │      + SonnetFailure    │                   │     loop2 = _retry_loop(    │
   │        Summary          │                   │         model=opus,         │
   └─────────────────────────┘                   │         attempt_offset=N)   │
                                                 └─────────────────────────────┘
```

**Two integration points, by design:**
1. Codegen failures are stage-boundary events (a `BudgetExhaustedError` exits the stage). Catch-and-restart at orchestrator level.
2. Stage 4 failures are loop-internal (the loop ends below_target without raising). Extend the same loop with a different model rather than restarting the stage.

This non-uniformity follows the natural shape of each stage's failure mode.

---

## 4. Components

### 4.1 `SynthesisConfig` additions

```python
# src/cuda_engine/config.py
class SynthesisConfig(BaseModel):
    # … existing fields …
    escalate_to_opus_on_bust: bool = True            # already exists
    opus_retry_budget_codegen: int = 3               # NEW
    opus_retry_budget_performance: int = 1           # NEW
    sonnet_model: str = "claude-sonnet-4-6"          # already exists
    opus_model: str = "claude-opus-4-7"              # already exists
```

`escalate_to_opus_on_bust=False` makes both escalation paths no-ops. Keeps existing tests unchanged where they don't want to mock Opus calls.

### 4.2 `BudgetExhaustedError` carries structured failure data

```python
# src/cuda_engine/stages/base.py
@dataclass(frozen=True)
class SonnetFailureSummary:
    last_compile_errors: str
    last_compile_log: str
    last_source_attempt: str
    attempts_made: int


class BudgetExhaustedError(RuntimeError):
    def __init__(self, message: str, summary: SonnetFailureSummary | None = None) -> None:
        super().__init__(message)
        self.summary = summary
```

Internal-only dataclass; never crosses the public API boundary, so no Pydantic.

### 4.3 `Stage2Codegen.run()` accepts `model` and `escalation_context`

```python
def run(
    self,
    *,
    spec: KernelSpec,
    run_id: str,
    model: str,                                       # NEW (replaces hardcoded string)
    retry_budget: int = 3,
    repair_context: CorrectnessReport | None = None,
    escalation_context: SonnetFailureSummary | None = None,   # NEW
    artifact_prefix: str = "stage2_codegen",
) -> KernelArtifact:
```

When `escalation_context` is set, the initial user prompt prepends a structured "previous Sonnet attempts failed" block listing the last compile errors, last log, and last source. This nudges Opus toward the unsolved problem instead of repeating Sonnet's path.

When the retry budget exhausts, `BudgetExhaustedError` is raised with a populated `SonnetFailureSummary` constructed from `messages` history and `last_result`.

Artifact prefix convention:
- Sonnet attempts: `stage2_codegen/attempt_NN/`
- Opus retry after escalation: `stage2_codegen/escalated/attempt_NN/`
- Repair-loop equivalent: `stage3_repair/attempt_KK/codegen/escalated/attempt_NN/`

### 4.4 Orchestrator helper `_run_codegen_with_escalation`

```python
# src/cuda_engine/orchestrator.py
def _run_codegen_with_escalation(
    *,
    llm: _TracingLLMClient,
    gpu: GPURunner,
    store: ArtifactStore,
    cfg: SynthesisConfig,
    run_args: dict[str, Any],          # spec, run_id, retry_budget, repair_context, artifact_prefix
) -> KernelArtifact:
    stage = Stage2Codegen(llm=llm, gpu=gpu, store=store)
    try:
        return stage.run(**run_args, model=cfg.sonnet_model)
    except BudgetExhaustedError as bust:
        if not cfg.escalate_to_opus_on_bust:
            raise
        opus_args = {
            **run_args,
            "retry_budget": cfg.opus_retry_budget_codegen,
            "artifact_prefix": f"{run_args['artifact_prefix']}/escalated",
            "escalation_context": bust.summary,
        }
        return stage.run(**opus_args, model=cfg.opus_model)
```

Both call sites in `Orchestrator.run` (initial codegen at line ~62 and the repair-loop `repair_action` at line ~96) route through this helper.

### 4.5 Stage 4 perf escalation

`_retry_loop` gains two parameters:

```python
def _retry_loop(
    self,
    *,
    spec, artifact, benchmark, speedup, target, inputs, run_id,
    retry_budget: int,
    model: str,                # NEW
    attempt_offset: int = 0,   # NEW (so Opus attempts continue numbering)
) -> tuple[…]:
```

`run()` calls it twice when escalation triggers:

```python
artifact, bench, speedup, warnings, notes = self._retry_loop(
    …, model=self.cfg.sonnet_model, retry_budget=retry_budget, attempt_offset=0,
)

if (
    speedup < target
    and self.cfg.escalate_to_opus_on_bust
    and self.cfg.opus_retry_budget_performance > 0
    and self.llm is not None
):
    artifact, bench, speedup, w2, n2 = self._retry_loop(
        …,
        model=self.cfg.opus_model,
        retry_budget=self.cfg.opus_retry_budget_performance,
        attempt_offset=retry_budget,
    )
    notes.append(
        f"escalated to opus after sonnet retry budget exhausted at speedup {speedup:.3f}"
    )
    warnings.extend(w2)
    notes.extend(n2)
```

The Opus loop receives the post-Sonnet artifact + benchmark + metrics as its starting state — the "structured Sonnet failure summary" is the loop's own input state. No extra summary type needed for Stage 4.

### 4.6 Stage refactor (parameterize hardcoded models)

| File | Line | Change |
|---|---|---|
| `src/cuda_engine/stages/codegen.py` | 52 | `model="claude-sonnet-4-6"` → `model=model` (new param) |
| `src/cuda_engine/stages/interview.py` | 51 | same — add `model: str` param |
| `src/cuda_engine/stages/polish.py` | 45 | same — add `model: str` param |

All three `.run()` methods gain `model: str` as a required keyword-only param (no default — orchestrator always passes one). Existing unit tests are updated to pass `model="claude-sonnet-4-6"` explicitly. Lands as its own commit before the escalation logic.

---

## 5. Data flow — escalation sequence

### Codegen escalation
```
1. Orchestrator → _run_codegen_with_escalation(args)
2. Stage2Codegen.run(model=sonnet_model)
3. Sonnet attempts 1..N all fail compile
4. BudgetExhaustedError raised with SonnetFailureSummary(last_compile_errors=…, last_source_attempt=…, attempts_made=N)
5. Wrapper catches, builds Opus args with escalation_context=summary, artifact_prefix=…/escalated
6. Stage2Codegen.run(model=opus_model, escalation_context=summary, retry_budget=opus_retry_budget_codegen)
7. Opus prompt prepends structured failure block, retries until success or its own budget exhausts
8. If Opus also busts → BudgetExhaustedError propagates uncaught (orchestrator surfaces stage failure as today)
```

### Stage 4 escalation
```
1. Stage4Performance.run() → benchmark
2. If below_target → _retry_loop(model=sonnet, retry_budget=N, offset=0)
3. Sonnet exhausts; final artifact/bench/metrics live in stage4_performance/perf_repair/attempt_NN/
4. run() checks: still below_target AND escalate enabled AND opus_retry_budget_performance > 0
5. _retry_loop(model=opus, retry_budget=M, offset=N) — Opus iterations write to attempt_(N+1)…(N+M)/
6. Final speedup/artifact/warnings/notes flow into PerformanceReport as today
7. Polish runs regardless (Stage 4 is soft-gate)
```

---

## 6. Trace and observability

`StageTrace.model_used` already aggregates distinct model names across all LLM calls in a stage via `_model_summary(responses)`. After escalation:
- Codegen trace `model_used`: `"claude-sonnet-4-6, claude-opus-4-7"` (multi-model run, no string change needed).
- Stage 4 trace `model_used`: same format.
- `attempts` field naturally counts both Sonnet and Opus LLM calls.

No `StageTrace` schema change. No new `"sonnet→opus"` literal — we let the existing aggregator handle multi-model runs uniformly.

`PerformanceReport.notes` gains an "escalated to opus after sonnet retry budget exhausted" entry when the Stage 4 escalation fires. CLI output unchanged.

---

## 7. Error handling

| Scenario | Behavior |
|---|---|
| Sonnet codegen busts, escalation disabled | `BudgetExhaustedError` propagates as today |
| Sonnet codegen busts, Opus succeeds | `KernelArtifact` returned, trace shows both models |
| Sonnet codegen busts, Opus also busts | `BudgetExhaustedError` from Opus call propagates (orchestrator routes to Stage-3 failure path as today) |
| Stage 4 below_target, escalation disabled | Same as today: soft fail with `below_target=True` |
| Stage 4 below_target, Opus hits target | `below_target=False`, `notes` records escalation |
| Stage 4 below_target, Opus also fails | `below_target=True`, warnings include Opus loop's warnings, `notes` records escalation |
| `escalation_context` passed to non-escalated codegen call | Ignored (no orchestrator path constructs this) — defensive: behaves identically to None |

---

## 8. Testing

### Unit (offline, fast)
- `tests/unit/stages/test_codegen.py`
  - `test_codegen_accepts_model_parameter` — `self.llm.complete(model=...)` receives passed value.
  - `test_codegen_uses_escalation_context_in_prompt` — initial user message contains failure-summary fields when context provided.
  - `test_codegen_raises_budget_exhausted_with_summary` — raised error has populated `SonnetFailureSummary`.
- `tests/unit/test_orchestrator.py`
  - `test_codegen_escalates_to_opus_on_bust` — MockLLMClient fails Sonnet 3× then succeeds 1st Opus attempt; `model_used` contains both, artifact path under `escalated/`.
  - `test_codegen_escalation_disabled_propagates_bust` — with `escalate_to_opus_on_bust=False`, orchestrator surfaces `BudgetExhaustedError` (or returns Stage-3 failure result, depending on call site).
  - `test_perf_escalation_runs_when_below_target` — Stage 4 Sonnet loop ends below_target; one Opus iteration runs, `notes` contains "escalated to opus", trace `model_used` includes Opus.
  - `test_perf_escalation_skipped_when_disabled` — same setup with escalation disabled; LLM call count == Sonnet attempts only.
- `tests/unit/stages/test_performance.py`
  - `test_retry_loop_accepts_model_and_offset` — calling `_retry_loop` twice (Sonnet then Opus) produces non-clobbering attempt dirs `01..N` then `N+1..N+M`.
  - `test_perf_skips_escalation_when_target_met` — Sonnet hits target; Opus loop never invoked.
- `tests/unit/test_config.py`
  - assert new config fields exist with documented defaults.

### Integration (Colab/A100, manual run)
- `tests/integration/test_e2e_perf_loop_escalation.py` — `@pytest.mark.integration`, skips on missing nvcc/ncu/CUDA/`ANTHROPIC_API_KEY`.
  - Kernel: contrived softmax / similar where the obvious implementation is well below 1.0× torch.compile and a competent optimization needs shared-memory tiling.
  - Config: `retry_budgets.performance=1`, `opus_retry_budget_performance=1` to bound wall time (~3-6 min on Colab).
  - Assertions:
    - At least one attempt directory exists under `stage4_performance/perf_repair/attempt_01/`.
    - At least one attempt directory exists at `attempt_02/` (the Opus iteration) with `nsight.json` + `benchmark.json`.
    - Performance stage trace `model_used` contains both `claude-sonnet-4-6` and `claude-opus-4-7`.
    - `result.report.warnings` contains the escalation note OR the kernel passed the target after Opus.
  - Soft assertion: don't require Opus to actually beat the target — just require it ran. Real-world variability is not a CI invariant.

### Verification
- Full unit suite: `python -m pytest -q --ignore=tests/integration` — all green.
- Lint: `ruff check src tests`.
- Types: `mypy src`.
- Integration test runs on Colab manually before declaring 4.3 done.

---

## 9. Commit plan (preview)

Detailed steps belong to the implementation plan. High-level commit sequence:

1. `refactor(stages): parameterize model on codegen/interview/polish` — mechanical refactor + test updates.
2. `feat(stages): SonnetFailureSummary + structured BudgetExhaustedError`.
3. `feat(config): add opus_retry_budget_codegen / opus_retry_budget_performance`.
4. `feat(orchestrator): codegen Sonnet→Opus escalation wrapper`.
5. `feat(stage4): Sonnet→Opus escalation in perf retry loop`.
6. `test(integration): Colab perf-loop escalation end-to-end`.

Each commit ships its own tests and stays passing. Aim: 6 commits, ~400 LOC + tests.

---

## 10. Open questions

None remaining at design time. Surfaces during implementation get logged in the plan.
