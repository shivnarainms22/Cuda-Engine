# M4 — v1.0 Release Gate Evidence

Branch: `m3/perf-loop`. Tracks the M4 checkpoint (design §6.2 / plan Task 5.x).
The honest baseline fix (`21f3b2b`) and the eval numbers it produced are the
backbone of this gate — see [M3-evidence.md](M3-evidence.md) for the fix detail.

## M4 Checkpoint Status

| Requirement | Status | Evidence |
|---|---|---|
| Internal regression ≥95% functional, median ≥1.0×, p25 ≥0.7×, fast_1 ≥30% | ✅ **MET** | Run `2026-06-01-191749`: 30/30 (100%), median 1.04×, p25 1.00×, fast_1 24/30 (80%). All vs the *fastest* torch.compile mode at N=16M. |
| KernelBench external subset ≥80% functional | 🟡 in progress | 12 hand-translated unseen in-scope fixtures (`9714c04`). Run `kernelbench-2026-06-02-015450`: 9/9 functional on kernels run; 3 (`reverse_cumsum`, `softplus`, `softsign`) stalled on credit exhaustion (last-3-consecutive `n/a` signature), not real failures. Resume after credit top-up → expect ≥10/12. |
| PR / nightly / pre-release CI all green | 🟡 partial | PR CI green through `9714c04` (175 unit tests, ruff, mypy). Nightly + pre-release `eval.yml` exist (M3/Task 5.8). |
| `pip install` from TestPyPI on fresh Colab works end-to-end | ❌ pending | Task 5.10 — needs tag + publish. Local wheel build dry-run done (see below). |
| Streamlit demo runs end-to-end | ✅ built | `examples/web_demo.py` (Task 5.3). |
| README: runnable quickstart, eval numbers, honest scope | ✅ MET | Updated 2026-06-01 with honest internal numbers + best-mode baseline note. |
| ≥3 worked examples in `examples/kernels/` | ✅ MET | `rmsnorm_silu_fp16`, `softmax_lastdim_fp16`, `topk_fp32`. |
| Privacy + cost docs | ✅ MET | `docs/privacy.md`, `docs/cost.md` (Task 5.7). |
| `evals/results/v1.0-<date>/` committed with release artifact | ❌ pending | Awaiting full KernelBench completion; commit internal + external summaries once the 3 stalled kernels finish. |

## Internal suite — gate cleared

See M3-evidence.md "Honest Baseline Re-run". 30/30 functional, clears every
perf axis of the original aggressive gate. No re-scope of the v1 perf bar was
needed once the baseline was measured honestly.

## KernelBench external subset

12 unseen, in-scope level1 ops hand-translated into `evals/kernelbench/filtered/`
(none overlap the internal 30). Per-kernel results from `kernelbench-2026-06-02-015450`:

```text
argmin_fp32          PASS  1.05
elu_fp16             PASS  1.05
frobenius_norm_fp32  PASS  1.17   (scalar output works)
l1_norm_fp32         PASS  0.94
leaky_relu_fp16      PASS  1.12
log_softmax_fp16     PASS  1.31
masked_cumsum_fp32   PASS  1.03   (2-input scan works)
mingpt_gelu_fp16     PASS  1.03
mse_loss_fp32        PASS  1.00   (2-input scalar reduction works)
reverse_cumsum_fp32  PENDING (credit exhaustion, not a real failure)
softplus_fp16        PENDING (credit exhaustion)
softsign_fp16        PENDING (credit exhaustion)
```

Functional: 9/9 of completed kernels. 7/9 strictly faster than torch.compile.
Recovery after top-up: `cuda-engine eval --suite kernelbench --out "$OUT_DIR"
--only reverse_cumsum_fp32,softplus_fp16,softsign_fp16 --no-resume --yes`, then a
plain `--resume` pass to regenerate the 12-row summary.

## Local wheel build dry-run (Task 5.10 de-risk, no network)

See the build/install evidence appended below — confirms the package builds and
imports in a clean environment before any TestPyPI publish.

## Remaining to ship v1.0

1. Top up credits → finish the 3 KernelBench kernels → confirm ≥10/12.
2. Commit `evals/results/v1.0-<date>/` with both summaries.
3. Tag `v1.0`, build wheel, publish to **TestPyPI**, validate fresh-Colab install (Task 5.10).
4. Merge `m3/perf-loop` → `main`.
