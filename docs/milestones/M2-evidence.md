# M2 Checkpoint Evidence

**Milestone:** M2 — Full Correctness Pipeline (Stages 1, 3, 5 real; hard gate enforced)
**Plan reference:** `docs/superpowers/plans/2026-04-26-cuda-synthesis-engine-plan.md` § 1589–1595
**Branch:** `m2/stage1-interview`
**Sign-off date:** _TBD — fill in once Colab artifacts are pasted below_
**Sign-off commit SHA:** _TBD_

---

## Checkpoint items

### 1. On Colab A100: 5/5 simplest kernels pass `passed=True`

- Target kernels: `vector_add_fp32`, `scalar_multiply_fp32`, `sum_reduction_fp32`, `argmax_fp32`, `rms_norm_fp16`.
- Evidence: pytest summary from the wrap-up Colab cell (see § Wrap-up Colab cell).
- Status: **PARTIAL** — each kernel has been confirmed individually; awaiting single-batch run.

```
PASTE pytest summary line here, e.g.:
============================== 5 passed in 348.21s ==============================
```

### 2. Hard gate verified on real GPU

Prompt: `"Generate a CUDA kernel for vector addition: out = x + y for fp32 tensors."`
Reference: `lambda x, y: 2 * x` (deliberately disagrees with prompt).
Config: `RetryBudgets(correctness=0)` so repair is disabled and Stage 3 fails immediately.
Test: `tests/integration/test_e2e_hard_gate_sad_path.py`.

Mock-level evidence already covered by `tests/unit/test_orchestrator.py::test_orchestrator_hard_gate_fails_on_correctness_mismatch` (asserts `failed_stage=3`).

```
PASTE excerpt of result.correctness.failing_inputs and report.json here:
{
  "passed": false,
  "failed_stage": 3,
  "correctness": { "passed": false, "max_abs_err": ..., "failing_inputs": [...] }
}
```

### 3. KernelSpec immutability

`tests/unit/models/test_models.py:30` asserts that mutating a frozen `KernelSpec` raises `pydantic.ValidationError`. All four model classes (`KernelSpec`, `KernelArtifact`, plus the inner `TensorArg` / `PrecisionTolerance`) carry `model_config = ConfigDict(frozen=True)` — see `src/cuda_engine/models/spec.py:19,28,37` and `src/cuda_engine/models/artifact.py:7`.

Status: **DONE** (unit-level evidence is sufficient — pydantic semantics do not depend on environment).

### 4. `report.stage_traces` populated for all 5 stages on a successful run

Wired in `src/cuda_engine/orchestrator.py` via `_TracingLLMClient` + `_run_traced_stage`; every call to a stage records a `StageTrace` with `attempts`, `succeeded`, `model_used`, `tokens_in/out`, `cache_read_tokens`, and `latency_seconds`. Aggregated totals (`total_llm_tokens_in/out`) appear on the top-level `SynthesisReport`.

Stages emitted on a happy-path run: `interview` → `codegen` → `correctness` (→ `codegen_repair` / `correctness` retry pairs if any) → `performance` → `polish`.

```
PASTE the stage_traces array from .test_artifacts/runs/<run_id>/report.json
(produced by `cuda-engine latest-report .test_artifacts/runs`).
Confirm: 5 distinct stage_name values, non-zero tokens_in on the LLM stages,
and at least one cache_read_tokens > 0 (prompt cache is wired).
```

### 5. Stage5Polish produces an annotated kernel; annotations explain tile/layout choice

Manual eyeball of `.test_artifacts/runs/<run_id>/stage5_polish/final/kernel.cu` (or whichever artifact path the polish stage writes). Confirm the annotated kernel includes inline comments justifying tile size, thread-block layout, dtype choice, and any memory-coalescing decisions.

```
PASTE 10–20 line excerpt from a polished kernel.cu showing a representative
annotation block. Note kernel name and which integration test produced it.
```

---

## Wrap-up Colab cell

Run this in a fresh Colab A100 session against the latest `m2/stage1-interview` HEAD. It runs all 6 integration tests (5 happy-path + 1 sad-path) in a single pytest invocation, prints the latest report summary, and tars the run dir for inspection.

```bash
%cd /content/Cuda-Engine
!git fetch --all && git checkout m2/stage1-interview && git pull
!pip install -e ".[dev]" -q
!rm -rf .test_artifacts/runs && mkdir -p .test_artifacts/runs

# Run all M2 integration tests in one batch.
!pytest tests/integration -v -m integration -s 2>&1 | tee .test_artifacts/m2-checkpoint.log

# Dump the most recent report so we can copy stage_traces/correctness into the evidence doc.
!cuda-engine latest-report .test_artifacts/runs

# Tarball the artifacts for offline inspection.
!tar -czf .test_artifacts/m2-checkpoint.tar.gz .test_artifacts/runs .test_artifacts/m2-checkpoint.log
!ls -lh .test_artifacts/m2-checkpoint.tar.gz
```

After the cell finishes, paste:
1. The pytest summary line into § 1.
2. The hard-gate run's `correctness` block from `report.json` into § 2.
3. One run's `stage_traces` array into § 4.
4. A representative polish annotation excerpt into § 5.
5. Update the **Sign-off commit SHA** at the top to `git rev-parse HEAD`.

---

## Out of scope for M2

The following appear in commits on this branch but belong to M3 — they are acceptable carry-over:

- `feat: add lightweight performance benchmarking` (`25f7e85`)
- `feat: configure performance benchmark settings` (`c05d92b`)
- `feat: show performance details in report cli` (`37cc397`)

These do not block the M2 sign-off; they will be hardened (Nsight, perf prompt, escalation) in M3.
