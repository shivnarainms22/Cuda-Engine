# M3 Task 4.3 Evidence

**Milestone:** M3 — Task 4.3, Sonnet->Opus escalation  
**Plan reference:** `docs/superpowers/plans/2026-05-07-stage-escalation-plan.md`  
**Branch:** `m3/perf-loop`  
**Status:** Signed off for Task 4.3 with accepted Colab A100 skip.

---

## Local Verification

Local verification was run on Windows against the current `m3/perf-loop` working tree after adding the missing orchestrator disabled-path coverage.

Commands:

```text
python -m pytest tests/unit/test_orchestrator.py::test_orchestrator_perf_stage_skips_opus_when_escalation_disabled -v
```

Result: `1 passed, 1 warning`.

```text
python -m pytest -q --ignore=tests/integration
```

Result: full non-integration suite passed with one existing Pydantic namespace warning.

```text
python -m ruff check src tests
```

Result: `All checks passed!`

```text
python -m mypy src
```

Result: `Success: no issues found in 36 source files`.

```text
python -m pytest tests/integration/test_e2e_perf_loop_escalation.py --collect-only -q
python -m pytest tests/integration/test_e2e_perf_loop_escalation.py -v
```

Result: one integration test collected; local run skipped because the local Windows environment does not provide the real Colab/A100/ncu/Anthropic combination.

---

## Plan Compliance Notes

- Stage 2 codegen escalation is implemented at orchestrator level via `_run_codegen_with_escalation`.
- Stage 4 performance escalation is implemented inside `Stage4Performance.run`.
- `SynthesisConfig` has `opus_retry_budget_codegen` and `opus_retry_budget_performance`.
- Unit coverage includes codegen escalation enabled/disabled, performance escalation enabled, and performance escalation disabled through both stage-level and orchestrator-level tests.
- The Colab runbook exists at `docs/milestones/M3-task-4.3-colab-runbook.md`.

---

## Colab A100 Checkpoint

Run date: 2026-05-10  
Environment: Colab Pro + A100 with real CUDA/ncu/Anthropic path  
Commit tested: `1ab3da625507f1eb1223d9cdf54f79b486f8fd13`  
Commit summary: `1ab3da6 (HEAD -> m3/perf-loop, origin/m3/perf-loop) docs(m3): Colab runbook for Task 4.3 perf-loop escalation test`

Command:

```text
python -m pytest tests/integration/test_e2e_perf_loop_escalation.py -v -s --tb=short -m integration
```

Result:

```text
=========================== short test summary info ============================
SKIPPED [1] tests/integration/test_e2e_perf_loop_escalation.py:72: escalation did not trigger (Sonnet hit target on first try): model_used=none
======================== 1 skipped in 263.95s (0:04:23) ========================
```

Interpretation: accepted skip per the Task 4.3 runbook. The performance stage did not call the LLM (`model_used=none`) because the initial kernel met the perf target before the Sonnet retry loop or Opus escalation was needed.

Notes:

- Sonnet and Opus perf-attempt speedups are not available for this run because the perf retry/escalation path was not entered.
- Final `below_target` was not printed in the provided pytest summary; the skip reason indicates the target was met before escalation.
