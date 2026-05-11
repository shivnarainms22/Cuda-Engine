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

---

## Internal Eval Checkpoint

Run date: 2026-05-10  
Environment: Colab Pro + A100 with real CUDA/ncu/Anthropic path  
Branch: `m3/perf-loop`  
Evidence type: console output supplied during Colab run. The Colab runtime reset before the `evals/results/2026-05-10` artifact directory could be zipped or downloaded, so this section records console-confirmed results rather than durable result artifacts.

Command:

```text
cuda-engine eval --suite internal --out evals/results/2026-05-10
```

The first resumable run completed all 30 kernels after five existing partial results were skipped. Summary output:

```text
Eval complete: 29/30 passed
CSV: evals/results/2026-05-10/results.csv
Summary: evals/results/2026-05-10/summary.md
```

Per-kernel summary from `summary.md`:

```text
Pass rate: 29/30

add_relu_fp32                 PASS  1.02
argmax_fp32                   PASS  1.00
bias_gelu_fp16                PASS  0.99
clamp_fp32                    PASS  1.00
cumulative_max_fp32           PASS  1.00
dropout_fp16                  PASS  1.00
geglu_fp16                    PASS  0.98
gelu_fp16                     PASS  1.00
l2_norm_fp32                  PASS  1.00
layernorm_fp16                PASS  1.00
layernorm_silu_fused_fp16     FAIL
masked_mean_fp16              PASS  1.00
max_lastdim_fp32              PASS  1.00
mean_lastdim_fp32             PASS  1.00
min_lastdim_fp32              PASS  1.00
prefix_sum_fp32               PASS  1.00
relu_bias_fp32                PASS  1.02
rms_norm_fp16                 PASS  1.00
rmsnorm_silu_fused_fp16       PASS  1.00
scalar_multiply_fp32          PASS  1.00
segment_sum_fp32              PASS  1.00
sigmoid_mul_fp16              PASS  0.77
silu_fp16                     PASS  1.00
softmax_lastdim_fp16          PASS  1.00
softmax_numerator_fp16        PASS  1.00
sum_reduction_fp32            PASS  1.00
swiglu_fp16                   PASS  1.01
tanh_add_fp32                 PASS  0.98
topk_fp32                     PASS  1.00
vector_add_fp32               PASS  1.10
```

The only failure was not a generated-kernel correctness issue. The failed per-kernel JSON reported:

```text
layernorm_silu_fused_fp16 failed
failure_reason: TypeError: reference() takes 1 positional argument but 3 were given
```

Root cause: `evals/internal/layernorm_silu_fused_fp16/prompt.txt` was ambiguous, so Stage 1 inferred affine LayerNorm inputs while `reference.py` intentionally accepted only `x`.

Fix: commit `dafa563` (`fix(evals): disambiguate layernorm silu fixture inputs`) clarified that the fixture has exactly one input tensor and no affine gamma/beta parameters.

Targeted rerun after the fix:

```text
[1/1] RUN layernorm_silu_fused_fp16
[1/1] DONE layernorm_silu_fused_fp16 passed=True speedup=1.00
Eval complete: 1/1 passed
```

Final interpreted internal functional result: `30/30` kernels passed after the fixture prompt fix and targeted rerun.

Performance notes from the 30-kernel console summary:

- `fast_1` from the pre-fix aggregate CSV: `4/30` (`add_relu_fp32`, `relu_bias_fp32`, `swiglu_fp16`, `vector_add_fp32` were visibly above 1.0x).
- `below_target` from the pre-fix aggregate CSV: `4/30` (`bias_gelu_fp16`, `geglu_fp16`, `sigmoid_mul_fp16`, `tanh_add_fp32`).
- The v1 fast_1 target (`>= 10/30`) is not met yet based on this first eval pass.
- The M3 checkpoint still needs durable eval artifacts and specific evidence for Nsight feedback improvement and Sonnet->Opus escalation in an eval run.

---

## Focused Perf Triage Checkpoint

Run date: 2026-05-11  
Environment: Colab Pro + A100 with real CUDA/ncu/Anthropic path  
Branch: `m3/perf-loop`  
Commit tested: `79faf97` or later  
Durable output directory: `/content/drive/MyDrive/cuda-engine-evals/2026-05-11-011901`  
Evidence type: Google Drive-backed eval output plus console summary supplied during Colab run.

Command:

```text
cuda-engine eval \
  --suite internal \
  --out /content/drive/MyDrive/cuda-engine-evals/2026-05-11-011901 \
  --only bias_gelu_fp16,geglu_fp16,sigmoid_mul_fp16,tanh_add_fp32 \
  --resume
```

Summary:

```text
# CUDA Engine Eval Summary

Pass rate: 4/4

## M3 Metrics

- Pass rate: 4/4 (100.0%)
- Median speedup vs torch.compile: 0.90x
- P25 speedup vs torch.compile: 0.80x
- fast_1 kernels (>1.0x): 0/4
- Below target kernels: 3/4

| Kernel | Status | Speedup vs torch.compile | Regression |
|---|---|---:|---|
| bias_gelu_fp16 | PASS | 0.99 |  |
| geglu_fp16 | PASS | 1.00 |  |
| sigmoid_mul_fp16 | PASS | 0.77 |  |
| tanh_add_fp32 | PASS | 0.81 |  |
```

Per-kernel focused result:

```text
bias_gelu_fp16 passed=true speedup=0.9871414615408 below_target=true
geglu_fp16 passed=true speedup=1.0 below_target=false
sigmoid_mul_fp16 passed=true speedup=0.774570988647599 below_target=true
tanh_add_fp32 passed=true speedup=0.8114478401527432 below_target=true
```

Nsight-backed repair evidence:

```text
bias_gelu_fp16:
  attempt_02 improved 0.953 -> 0.955, nsight=true
  attempt_03 improved 0.955 -> 0.9871, nsight=true

geglu_fp16:
  attempt_01 improved 0.9691 -> 1.0, nsight=true

sigmoid_mul_fp16:
  attempt_01 improved 0.7432 -> 0.7492, nsight=true
  attempt_02 improved 0.7492 -> 0.757, nsight=true
  attempt_04 improved 0.7267 -> 0.7746, nsight=true

tanh_add_fp32:
  attempt_01 improved 0.8098 -> 0.8302, nsight=true
  attempt_03 improved 0.796 -> 0.8249, nsight=true
```

Interpretation:

- The focused rerun is functionally green (`4/4`).
- The M3 Nsight-feedback checkpoint is satisfied: multiple kernels show attempt-to-attempt speedup improvement with persisted `nsight.json` evidence.
- `geglu_fp16` recovered to target in the focused rerun.
- `bias_gelu_fp16`, `sigmoid_mul_fp16`, and `tanh_add_fp32` remain below target.
- The full M3 `fast_1 >= 10/30` checkpoint is still open; this focused subset has `0/4` fast_1 because `geglu_fp16` landed exactly at `1.00x`, and fast_1 is strictly `>1.0x`.

---

## Near-Threshold Perf Triage Checkpoint

Run date: 2026-05-11  
Environment: Colab Pro + A100 with real CUDA/ncu/Anthropic path  
Branch: `m3/perf-loop`  
Commit tested: `a7eee98` or later  
Durable output directory: `/content/drive/MyDrive/cuda-engine-evals/2026-05-11-011901`  
Durable zip artifact: `/content/drive/MyDrive/cuda-engine-evals/internal-eval-near-threshold-2026-05-11-011901.zip` (`45M`)  
Evidence type: Google Drive-backed eval output, zipped artifact, and console summary supplied during Colab run.

Command:

```text
cuda-engine eval \
  --suite internal \
  --out /content/drive/MyDrive/cuda-engine-evals/2026-05-11-011901 \
  --only layernorm_fp16,masked_mean_fp16,rms_norm_fp16,rmsnorm_silu_fused_fp16,softmax_lastdim_fp16,softmax_numerator_fp16 \
  --resume
```

Progress:

```text
[1/6] RUN layernorm_fp16
[1/6] DONE layernorm_fp16 passed=True speedup=1.00
[2/6] RUN masked_mean_fp16
[2/6] DONE masked_mean_fp16 passed=True speedup=1.00
[3/6] RUN rms_norm_fp16
[3/6] DONE rms_norm_fp16 passed=True speedup=1.00
[4/6] RUN rmsnorm_silu_fused_fp16
[4/6] DONE rmsnorm_silu_fused_fp16 passed=True speedup=1.00
[5/6] RUN softmax_lastdim_fp16
[5/6] DONE softmax_lastdim_fp16 passed=True speedup=1.00
[6/6] RUN softmax_numerator_fp16
[6/6] DONE softmax_numerator_fp16 passed=True speedup=1.00
Eval complete: 6/6 passed
```

Summary:

```text
# CUDA Engine Eval Summary

Pass rate: 6/6

## M3 Metrics

- Pass rate: 6/6 (100.0%)
- Median speedup vs torch.compile: 1.00x
- P25 speedup vs torch.compile: 1.00x
- fast_1 kernels (>1.0x): 0/6
- Below target kernels: 0/6

| Kernel | Status | Speedup vs torch.compile | Regression |
|---|---|---:|---|
| layernorm_fp16 | PASS | 1.00 |  |
| masked_mean_fp16 | PASS | 1.00 |  |
| rms_norm_fp16 | PASS | 1.00 |  |
| rmsnorm_silu_fused_fp16 | PASS | 1.00 |  |
| softmax_lastdim_fp16 | PASS | 1.00 |  |
| softmax_numerator_fp16 | PASS | 1.00 |  |
```

Exact CSV rows:

```text
layernorm_fp16 passed=true speedup=1.00 below_target=false
masked_mean_fp16 passed=true speedup=1.00 below_target=false
rms_norm_fp16 passed=true speedup=1.00 below_target=false
rmsnorm_silu_fused_fp16 passed=true speedup=1.00 below_target=false
softmax_lastdim_fp16 passed=true speedup=1.00 below_target=false
softmax_numerator_fp16 passed=true speedup=1.00 below_target=false
```

Interpretation:

- The near-threshold rerun is functionally green (`6/6`).
- No near-threshold kernel is below target.
- This batch does not improve `fast_1`; all six landed at `1.00x`, while fast_1 requires strictly `>1.0x`.
- The full M3 `fast_1 >= 10/30` checkpoint remains open.

---

## Full Resumed Eval Checkpoint

Run date: 2026-05-11  
Environment: Colab Pro + A100 with real CUDA/ncu/Anthropic path  
Branch: `m3/perf-loop`  
Commit tested: `a7eee98` or later  
Durable output directory: `/content/drive/MyDrive/cuda-engine-evals/2026-05-11-011901`  
Durable zip artifact: `/content/drive/MyDrive/cuda-engine-evals/internal-eval-full-2026-05-11-011901.zip` (`114M`)  
Evidence type: Google Drive-backed eval output, zipped artifact, and console summary supplied during Colab run.

Command:

```text
cuda-engine eval \
  --suite internal \
  --out /content/drive/MyDrive/cuda-engine-evals/2026-05-11-011901 \
  --resume
```

Summary:

```text
# CUDA Engine Eval Summary

Pass rate: 28/30

## M3 Metrics

- Pass rate: 28/30 (93.3%)
- Median speedup vs torch.compile: 1.00x
- P25 speedup vs torch.compile: 1.00x
- fast_1 kernels (>1.0x): 1/30
- Below target kernels: 5/30
```

Failures:

```text
topk_fp32:
  failure_reason: BadRequestError: Error code: 400 - credit balance is too low to access the Anthropic API
  report.json: missing
  artifacts persisted before failure:
    inputs/config.json
    inputs/prompt.txt
    inputs/reference.py
    stage1_interview/prompt_to_llm.md

vector_add_fp32:
  failure_reason: BadRequestError: Error code: 400 - credit balance is too low to access the Anthropic API
  report.json: missing
  artifacts persisted before failure:
    inputs/config.json
    inputs/prompt.txt
    inputs/reference.py
    stage1_interview/prompt_to_llm.md
```

Interpretation:

- The full resumed run produced durable artifacts and a durable zip.
- `28/30` kernels passed in this run; the M3 functional threshold (`>=25/30`) is met even counting the two API-credit failures as failures.
- The two failing rows are external API-credit failures during Stage 1, before generated kernels or correctness/performance execution. They are not evidence of kernel correctness failures.
- `topk_fp32` and `vector_add_fp32` need a targeted `--no-resume` rerun after Anthropic credits are restored, because their failed per-kernel JSON files now exist and plain `--resume` would skip them.
- `fast_1 >= 10/30` remains open (`1/30` in this run).

---

## GitHub CI Checkpoint

Run date: 2026-05-11  
Branch: `m3/perf-loop`  
Fix commit: `bf9286f` (`fix(ci): make evals importable in unit tests`)  
Passing workflow run: `25650538197`  
Workflow: `PR`

Issue: earlier GitHub `PR` workflow runs failed during unit-test collection with:

```text
ModuleNotFoundError: No module named 'evals'
```

Affected tests:

```text
tests/unit/test_cli.py
tests/unit/test_internal_eval_suite.py
```

Fix:

- Added `pythonpath = ["."]` to `[tool.pytest.ini_options]` in `pyproject.toml`.
- Added `evals/__init__.py` so the source-checkout eval runner is importable as a package in CI.

Verification:

```text
GitHub run 25650538197 completed successfully:
- ruff check src tests
- mypy src/
- pytest tests/unit -v --cov=cuda_engine --cov-report=term-missing -m "not integration"
```

Local CI-parity verification also passed:

```text
python -m pytest tests/unit -v --cov=cuda_engine --cov-report=term-missing -m "not integration"
python -m ruff check src tests evals
python -m mypy src
```

---

## M3 Checkpoint Status

| Requirement | Status | Evidence |
|---|---|---|
| All 30 internal kernels run end-to-end via `cuda-engine eval --suite internal` | Partial | Full resumed run attempted all 30. Two rows (`topk_fp32`, `vector_add_fp32`) stopped during Stage 1 due Anthropic credit exhaustion, not CUDA failure. Rerun only those two with `--no-resume` after credits are restored. |
| Pass rate >= 25/30 functional on Colab A100 | Met | Full resumed run: `28/30` passed, even counting two external API-credit failures as failures. Earlier console-confirmed run plus targeted fixture rerun interpreted as `30/30` functional. |
| >= 10/30 hit fast_1 (`>1.0x` vs torch.compile median) | Open | Full resumed run: `fast_1=1/30`; previous console-confirmed run showed `4/30`. Needs targeted performance strategy before M4. |
| At least one kernel demonstrates Nsight feedback improvement | Met | Focused perf triage showed multiple `nsight=true` attempt improvements, including `geglu_fp16` `0.9691 -> 1.0` and `bias_gelu_fp16` `0.955 -> 0.9871`. |
| At least one kernel demonstrates Sonnet->Opus escalation | Partially satisfied by accepted runbook skip | Colab Task 4.3 integration skipped because the initial kernel hit target before escalation (`model_used=none`). This is accepted by the Task 4.3 runbook, but there is not yet a real eval-row example where Opus succeeds after Sonnet bust. |
| Eval report contains aggregate markdown, per-kernel JSON, and CSV | Met | Drive-backed output at `/content/drive/MyDrive/cuda-engine-evals/2026-05-11-011901`; full zip `internal-eval-full-2026-05-11-011901.zip` (`114M`). |
| Nightly CI workflow exists | Met | `.github/workflows/nightly.yml` added in `78710cd`, configured for self-hosted A100 runner labels and artifact upload. |

Current M3 blockers before moving to M4:

1. Restore Anthropic credits and rerun:

   ```text
   cuda-engine eval --suite internal --out "$OUT_DIR" --only topk_fp32,vector_add_fp32 --no-resume
   ```

2. Raise `fast_1` from current `1/30` toward the `>=10/30` checkpoint.
