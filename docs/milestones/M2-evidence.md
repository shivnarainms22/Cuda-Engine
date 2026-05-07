# M2 Checkpoint Evidence

**Milestone:** M2 — Full Correctness Pipeline (Stages 1, 3, 5 real; hard gate enforced)
**Plan reference:** `docs/superpowers/plans/2026-04-26-cuda-synthesis-engine-plan.md` § 1589–1595
**Branch:** `m2/stage1-interview`
**Sign-off date:** 2026-05-07
**Sign-off commit SHA:** `4174287a36f5cccead61b9ff9ab4b9b856f05afb`
**Hardware:** Colab A100 (sm_80)
**Status:** ✅ **PASSED**

---

## Checkpoint items

### 1. On Colab A100: 5/5 simplest kernels pass `passed=True` ✅

Target kernels: `vector_add_fp32`, `scalar_multiply_fp32`, `sum_reduction_fp32`, `argmax_fp32`, `rms_norm_fp16`.

All 8 integration tests passed in a single batch (`pytest tests/integration -v -m integration -s`):

```
============================== 8 passed in <wall-time> ==============================
```

Run-id ↔ kernel mapping (from `report.json[*].report.spec_name`):

| run_id        | spec_name                |
| ------------- | ------------------------ |
| 5e5355e49ba3  | vector_add_fp32          |
| 1ac28b11cb46  | scalar_multiply_fp32     |
| 07e5bccc3b71  | row_sum_reduce_fp32      |
| 35512f0a03a8  | fp32_argmax_last_dim     |
| 491c6e074c88  | argmax_lastdim_fp32      |
| 4a6e23a298a3  | rms_norm_fp16_no_gamma   |
| 6369227276ae  | rms_norm_fp16_last_dim   |
| 30101e2e89c7  | (sad-path; passed=false) |

> Note: 7 happy-path run dirs for 5 happy-path tests (two argmax, two rms_norm). Likely Stage 1 retries producing distinct `spec.name` values within one synthesize call. Tracked as a follow-up — see § Follow-ups for M3.

### 2. Hard gate verified on real GPU ✅

Prompt: `"Generate a CUDA kernel for vector addition: out = x + y for fp32 tensors."`
Reference: `lambda x, y: 2 * x` (deliberately disagrees with prompt).
Config: `RetryBudgets(correctness=0)` — repair disabled so Stage 3 fails immediately.
Test: `tests/integration/test_e2e_hard_gate_sad_path.py` (real-GPU evidence).
Mock evidence: `tests/unit/test_orchestrator.py::test_orchestrator_hard_gate_fails_on_correctness_mismatch`.

Sad-path run (`30101e2e89c7`) `report.json` excerpt:

```python
{'passed': False,
 'failed_stage': 3,
 'failure_reason': 'correctness check failed',
 'correctness': {'passed': False,
                 'max_abs_err': 1.0,
                 'max_rel_err': 999999995904.0,
                 'shapes_tested': [[0], [1], [127], [128], [1024], [4097]],
                 'shape_results': [{'shape': [0],    'passed': True,  'max_abs_err': 0.0, 'max_rel_err': 0.0},
                                   {'shape': [1],    'passed': False, 'max_abs_err': 1.0, 'max_rel_err': 999999995904.0},
                                   {'shape': [127],  'passed': False, 'max_abs_err': 1.0, 'max_rel_err': 999999995904.0},
                                   {'shape': [128],  'passed': False, 'max_abs_err': 1.0, 'max_rel_err': 999999995904.0},
                                   {'shape': [1024], 'passed': False, 'max_abs_err': 1.0, 'max_rel_err': 999999995904.0},
                                   {'shape': [4097], 'passed': False, 'max_abs_err': 1.0, 'max_rel_err': 999999995904.0}],
                 'failing_inputs': [{'shape': [1],    'output_index': 0, 'max_abs_err': 1.0, 'max_rel_err': 999999995904.0},
                                    {'shape': [127],  'output_index': 0, 'max_abs_err': 1.0, 'max_rel_err': 999999995904.0},
                                    {'shape': [128],  'output_index': 0, 'max_abs_err': 1.0, 'max_rel_err': 999999995904.0},
                                    {'shape': [1024], 'output_index': 0, 'max_abs_err': 1.0, 'max_rel_err': 999999995904.0},
                                    {'shape': [4097], 'output_index': 0, 'max_abs_err': 1.0, 'max_rel_err': 999999995904.0}]}}
```

Shape `[0]` correctly passes (empty input, zero work). All non-empty shapes show consistent `max_abs_err=1.0`, matching the expected divergence between `x+y` (kernel) and `2*x` (reference) on the seeded inputs. `failing_inputs` is non-empty as required.

### 3. KernelSpec immutability ✅

`tests/unit/models/test_models.py:30` asserts that mutating a frozen `KernelSpec` raises `pydantic.ValidationError`. All four model classes (`KernelSpec`, `KernelArtifact`, plus the inner `TensorArg` / `PrecisionTolerance`) carry `model_config = ConfigDict(frozen=True)` — see `src/cuda_engine/models/spec.py:19,28,37` and `src/cuda_engine/models/artifact.py:7`.

Unit-level evidence is sufficient — pydantic frozen-model semantics are environment-independent.

### 4. `report.stage_traces` populated for all 5 stages on a successful run ✅

Wired in `src/cuda_engine/orchestrator.py` via `_TracingLLMClient` + `_run_traced_stage`. Every stage call records a `StageTrace` with `attempts`, `succeeded`, `model_used`, `tokens_in/out`, `cache_read_tokens`, and `latency_seconds`. Aggregated totals appear on the top-level `SynthesisReport`.

`stage_traces` from happy-path run `07e5bccc3b71` (row_sum_reduce_fp32):

```python
[{'stage_name': 'interview',    'attempts': 1, 'succeeded': True, 'model_used': 'claude-opus-4-7',  'tokens_in': 811,  'tokens_out': 263,  'cache_read_tokens': 0, 'latency_seconds':  3.349},
 {'stage_name': 'codegen',      'attempts': 1, 'succeeded': True, 'model_used': 'claude-sonnet-4-6','tokens_in': 1689, 'tokens_out': 2786, 'cache_read_tokens': 0, 'latency_seconds': 33.453},
 {'stage_name': 'correctness',  'attempts': 1, 'succeeded': True, 'model_used': 'none',             'tokens_in': 0,    'tokens_out': 0,    'cache_read_tokens': 0, 'latency_seconds': 10.489},
 {'stage_name': 'performance',  'attempts': 1, 'succeeded': True, 'model_used': 'none',             'tokens_in': 0,    'tokens_out': 0,    'cache_read_tokens': 0, 'latency_seconds':  2.625},
 {'stage_name': 'polish',       'attempts': 1, 'succeeded': True, 'model_used': 'claude-sonnet-4-6','tokens_in': 2156, 'tokens_out': 3022, 'cache_read_tokens': 0, 'latency_seconds': 58.117}]
```

All 5 stages emitted, all succeeded, latencies populated, token counts populated on LLM-driven stages, `model_used="none"` correctly recorded on stages that don't call the LLM (correctness, performance benchmarking).

> Two follow-ups (non-blocking): (a) `cache_read_tokens=0` everywhere — expected on a cold session, but needs verification on a warm rerun; (b) **interview ran on Opus while codegen/polish used Sonnet**, inverting the locked decision (Sonnet default + Opus escalation). Both tracked in § Follow-ups.

### 5. Stage5Polish produces an annotated kernel; annotations explain tile/layout choice ✅

Source path: `src/cuda_engine/stages/polish.py:101` writes the polished kernel to `<run_dir>/stage5_polish/final/kernel.cu`.

Excerpt from `.test_artifacts/runs/07e5bccc3b71/stage5_polish/final/kernel.cu` (row_sum_reduce_fp32):

```cuda
//
// Computes out[b] = sum_{d=0}^{D-1} x[b, d]  for a row-major fp32 tensor
// of shape [B, D].
//
// Target architecture : sm_80 (Ampere)
// Precision tolerance : rtol=0.001, atol=0.001  (verified max_abs_err = 0.0)
// Throughput          : ~12.96 GB/s (matches or ties torch.compile reference)
//
// Correctness summary
// -------------------
//   All tested shapes ([1,1], [2,3], [4,127], [8,256]) pass with 0 error.
//   fp32 accumulation is used throughout, matching the spec requirement.
//
// Performance summary
// -------------------
//   Speedup vs reference    : 1.0×  (on-par; bandwidth-bound at ~13 GB/s)
//   Spill bytes             : 0     (register pressure is low)
//   Occupancy               : not profiled, but 256-thread blocks with
//                             minimal shared memory (32 floats = 128 B) allow
//                             many concurrent blocks per SM.
//   The kernel is pure memory-bandwidth bound for all practical (B, D) sizes;
//   arithmetic intensity ≈ 1 ADD per 4 bytes read.

__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

// row_sum_reduce_kernel<BLOCK_DIM>
//
// Launch configuration
//   Grid  : dim3(B)          — one thread-block per input row
//   Block : dim3(BLOCK_DIM)  — 256 threads (= 8 warps); template parameter
```

Annotations explain: target SM, tolerance vs measured error, throughput, correctness coverage, dtype-accumulation choice, perf vs reference, register pressure, occupancy reasoning, arithmetic intensity, warp-reduce step pattern, launch grid/block rationale. Eyeball check satisfied.

---

## Follow-ups (deferred to M3 prep — non-blocking for M2 sign-off)

1. **Stage 1 ran on Opus instead of Sonnet.** `claude-opus-4-7` was used for the interview stage in the row_sum_reduce trace, while codegen + polish used Sonnet. The locked decision is Sonnet default + Opus on escalation. Either Stage 1 hardcodes Opus or escalation always fires. Investigate `src/cuda_engine/stages/interview.py` and the orchestrator's escalation hook before M3 begins.
2. **Duplicate happy-path run dirs.** 7 happy-path runs for 5 happy-path tests (two `argmax_*` variants, two `rms_norm_*` variants). M3's `evals/runner.py` assumes one run per `synthesize()` call. Determine whether retries or test fixtures produce the duplicates.
3. **Orphan run dir `4a687aa02162`** has no parseable `report.json`. Likely an aborted run from a Stage 1/2 crash before persistence. Add a defensive write of a minimal `report.json` on early failure paths.
4. **Prompt cache verification.** `cache_read_tokens=0` across all stages on this cold-session run. Verify on a warm session (back-to-back synthesize calls in one process) that `cache_read_tokens > 0` on the second call. If not, the `cache_control` headers on system blocks are not taking effect.

## Out of scope for M2 (acceptable carry-over to M3)

These commits exist on this branch but cover M3 territory:

- `feat: add lightweight performance benchmarking` (`25f7e85`)
- `feat: configure performance benchmark settings` (`c05d92b`)
- `feat: show performance details in report cli` (`37cc397`)

They will be hardened (Nsight, perf prompt, Sonnet→Opus escalation, 30-kernel eval) during M3.

---

## How to reproduce

The single Colab cell that produced this evidence:

```bash
%cd /content/Cuda-Engine
!git fetch --all && git checkout m2/stage1-interview && git pull
!pip install -e ".[dev]" -q
!rm -rf .test_artifacts/runs && mkdir -p .test_artifacts/runs
!pytest tests/integration -v -m integration -s 2>&1 | tee .test_artifacts/m2-checkpoint.log
!cuda-engine latest-report .test_artifacts/runs
!tar -czf .test_artifacts/m2-checkpoint.tar.gz .test_artifacts/runs .test_artifacts/m2-checkpoint.log
!git rev-parse HEAD
```
