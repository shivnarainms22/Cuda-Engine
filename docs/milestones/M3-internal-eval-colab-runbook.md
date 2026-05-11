# M3 Internal Eval Colab Runbook

**Branch:** `m3/perf-loop`  
**Command:** `cuda-engine eval --suite internal`  
**Hardware:** Colab Pro + A100 (`sm_80`) with `nvcc`, `ncu`, and `ANTHROPIC_API_KEY`  
**Persistence:** write `--out` under Google Drive and zip after each focused batch.

This runbook is optimized for M3 perf triage. It avoids writing long-run eval artifacts only to `/content`, because Colab runtime resets can delete `/content` before the result directory is downloaded.

---

## Cell 1 - Verify GPU Tooling

```bash
!nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
!which nvcc && nvcc --version | tail -1
!which ncu && ncu --version | head -1
```

Expected: A100, CUDA 12.x, and `ncu` on PATH.

---

## Cell 2 - Mount Drive And Clone

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
%cd /content
!rm -rf Cuda-Engine
!git clone --branch m3/perf-loop --depth 20 https://github.com/shivnarainms22/Cuda-Engine.git
%cd /content/Cuda-Engine
!pip install -e . --quiet
!pip install pytest pyyaml --quiet
!git rev-parse HEAD
!cuda-engine --help
```

---

## Cell 3 - Set API Key

```python
import os
from google.colab import userdata

os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
assert os.environ["ANTHROPIC_API_KEY"].startswith("sk-")
```

---

## Cell 4 - Define Durable Output Paths

```bash
export EVAL_RUN_ID="$(date +%Y-%m-%d-%H%M%S)"
export DRIVE_ROOT="/content/drive/MyDrive/cuda-engine-evals"
export OUT_DIR="$DRIVE_ROOT/$EVAL_RUN_ID"
export LOG_DIR="$DRIVE_ROOT/logs"
cat >/content/cuda-engine-eval-env.sh <<EOF
export EVAL_RUN_ID="$EVAL_RUN_ID"
export DRIVE_ROOT="$DRIVE_ROOT"
export OUT_DIR="$OUT_DIR"
export LOG_DIR="$LOG_DIR"
EOF
source /content/cuda-engine-eval-env.sh
mkdir -p "$OUT_DIR" "$LOG_DIR"
echo "OUT_DIR=$OUT_DIR"
```

---

## Cell 5 - Focused Below-Target Batch

Run the known below-target kernels first. This is the fastest useful rerun for M3 perf triage.

```bash
source /content/cuda-engine-eval-env.sh
cd /content/Cuda-Engine
cuda-engine eval \
  --suite internal \
  --out "$OUT_DIR" \
  --only bias_gelu_fp16,geglu_fp16,sigmoid_mul_fp16,tanh_add_fp32 \
  --resume \
  2>&1 | tee "$LOG_DIR/internal-eval-focused-$EVAL_RUN_ID.log"
```

Checkpoint the focused batch immediately:

```bash
source /content/cuda-engine-eval-env.sh
cd "$DRIVE_ROOT"
zip -qr "internal-eval-focused-$EVAL_RUN_ID.zip" "$EVAL_RUN_ID" "logs/internal-eval-focused-$EVAL_RUN_ID.log"
ls -lh "internal-eval-focused-$EVAL_RUN_ID.zip"
```

---

## Cell 6 - Inspect Focused Results

```bash
source /content/cuda-engine-eval-env.sh
python - <<'PY'
import csv
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
csv_path = out_dir / "results.csv"
print("csv:", csv_path)
with csv_path.open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        print(
            row["kernel"],
            "passed=", row["passed"],
            "failure_kind=", row.get("failure_kind", ""),
            "speedup=", row["speedup_vs_torch_compile"],
            "below_target=", row["below_target"],
            "artifacts=", row["artifacts_dir"],
        )

print("\nAttempt evidence:")
for json_path in sorted((out_dir / "kernels").glob("*.json")):
    row = json.loads(json_path.read_text())
    artifacts = Path(row["artifacts_dir"])
    perf_dir = artifacts / "stage4_performance" / "perf_repair"
    attempts = sorted(perf_dir.glob("attempt_*")) if perf_dir.exists() else []
    print(row["kernel"], "attempts=", [attempt.name for attempt in attempts])
PY
```

For the M3 Nsight-feedback checkpoint, preserve at least one kernel where an attempt has both `benchmark.json` and `benchmark_after.json`, and the after speedup improves over the before speedup.

`failure_kind` interpretation:

- Empty value: passed kernel or no failure recorded.
- `stage_failure`: the CUDA synthesis pipeline reached a stage failure, such as correctness or performance.
- `external_error`: an external service/environment problem, such as Anthropic API credit exhaustion or rate limiting.
- `runner_error`: an exception outside a normal stage failure that is not clearly external.

---

## Cell 7 - Optional Near-Threshold Batch

If the focused batch is stable, run near-threshold kernels to try to raise `fast_1`.

```bash
source /content/cuda-engine-eval-env.sh
cd /content/Cuda-Engine
cuda-engine eval \
  --suite internal \
  --out "$OUT_DIR" \
  --only layernorm_fp16,masked_mean_fp16,rms_norm_fp16,rmsnorm_silu_fused_fp16,softmax_lastdim_fp16,softmax_numerator_fp16 \
  --resume \
  2>&1 | tee "$LOG_DIR/internal-eval-near-threshold-$EVAL_RUN_ID.log"
```

Checkpoint again:

```bash
source /content/cuda-engine-eval-env.sh
cd "$DRIVE_ROOT"
zip -qr "internal-eval-near-threshold-$EVAL_RUN_ID.zip" "$EVAL_RUN_ID" "logs/internal-eval-near-threshold-$EVAL_RUN_ID.log"
ls -lh "internal-eval-near-threshold-$EVAL_RUN_ID.zip"
```

---

## Cell 8 - Full Suite Rerun When Ready

Only run this after focused triage is useful, because it can take hours.

```bash
source /content/cuda-engine-eval-env.sh
cd /content/Cuda-Engine
cuda-engine eval \
  --suite internal \
  --out "$OUT_DIR" \
  --resume \
  2>&1 | tee "$LOG_DIR/internal-eval-full-$EVAL_RUN_ID.log"
```

Final checkpoint:

```bash
source /content/cuda-engine-eval-env.sh
cd "$DRIVE_ROOT"
zip -qr "internal-eval-full-$EVAL_RUN_ID.zip" "$EVAL_RUN_ID" "logs/internal-eval-full-$EVAL_RUN_ID.log"
ls -lh "internal-eval-full-$EVAL_RUN_ID.zip"
```

---

## Cell 9 - Rerun Failed-Only Kernels

Use this only after inspecting `results.csv`. If a failed row already has a JSON file under `kernels/`, plain `--resume` will skip it. Use `--no-resume` when intentionally replacing failed per-kernel JSON.

Example for API-credit failures after credits are restored:

```bash
source /content/cuda-engine-eval-env.sh
cd /content/Cuda-Engine
cuda-engine eval \
  --suite internal \
  --out "$OUT_DIR" \
  --only topk_fp32,vector_add_fp32 \
  --no-resume \
  2>&1 | tee "$LOG_DIR/internal-eval-failed-only-$EVAL_RUN_ID.log"
```

Checkpoint immediately:

```bash
source /content/cuda-engine-eval-env.sh
cd "$DRIVE_ROOT"
zip -qr "internal-eval-failed-only-$EVAL_RUN_ID.zip" "$EVAL_RUN_ID" "logs/internal-eval-failed-only-$EVAL_RUN_ID.log"
ls -lh "internal-eval-failed-only-$EVAL_RUN_ID.zip"
```

---

## Evidence To Copy Back

Record these in `docs/milestones/M3-evidence.md` after the run:

- Commit SHA from `git rev-parse HEAD`.
- `summary.md` M3 metrics: pass rate, median speedup, p25 speedup, fast_1 count, below-target count.
- Focused kernel rows from `results.csv`.
- Any `failure_kind=external_error` rows and their exact external reason.
- At least one `stage4_performance/perf_repair/attempt_*/benchmark.json` to `benchmark_after.json` improvement.
- Zip filename and Drive path for durable artifacts.
