# Task 4.3 — Colab Runbook (Sonnet→Opus Escalation)

**Branch:** `m3/perf-loop` (pushed at `9d2d2d7`)
**Test:** `tests/integration/test_e2e_perf_loop_escalation.py`
**Hardware:** Colab Pro + A100 (sm_80) with `ncu` available
**Wall-time budget:** ~5–10 min (1 codegen + correctness + perf Sonnet attempt + perf Opus attempt + polish)

This runbook is a copy-paste sequence of Colab cells. Each cell is a separate block.

---

## Cell 1 — Verify A100 + ncu

```bash
!nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
!which nvcc && nvcc --version | tail -1
!which ncu && ncu --version | head -1
```

**Expected:** `NVIDIA A100-SXM4-...`, nvcc 12.x, ncu present. If `ncu` is missing, the Pro runtime sometimes ships without it — `apt-get install -y nvidia-cuda-toolkit-gcc` or use the `cuda-nsight-compute` package. Test will skip with a clear message if absent.

---

## Cell 2 — Clone branch and install

```bash
%cd /content
!rm -rf Cuda-Engine
!git clone --branch m3/perf-loop --depth 5 https://github.com/shivnarainms22/Cuda-Engine.git
%cd Cuda-Engine
!pip install -e . --quiet
!pip install pytest --quiet
```

---

## Cell 3 — Set API key

```python
import os
from google.colab import userdata
os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
assert os.environ["ANTHROPIC_API_KEY"].startswith("sk-")
```

If `ANTHROPIC_API_KEY` is not in Colab secrets, set it once via the key-icon sidebar.

---

## Cell 4 — Run the escalation integration test

```bash
!cd /content/Cuda-Engine && python -m pytest \
    tests/integration/test_e2e_perf_loop_escalation.py \
    -v -s --tb=short -m integration 2>&1 | tee /content/escalation-test.log
```

The `-s` flag streams synthesize() output so you can watch Stage 4 progress. Expect one of:

- ✅ **PASS** — escalation triggered and assertions all hold. Look for `escalated to opus after sonnet retry budget exhausted` in the log.
- ⚠️ **SKIPPED with reason "escalation did not trigger"** — Sonnet hit the perf bar on first try. Acceptable variance; not a defect. Re-run once or twice; if it consistently skips, the chosen workload (softmax) isn't slow enough on this Colab A100. Bump `performance_shape_n` or pick a more contrived kernel.
- ❌ **FAIL** — see "if it fails" below.

---

## Cell 5 — Inspect run artifacts on pass

```bash
RUN_DIR=$(ls -td /tmp/pytest-of-root/pytest-*/test_perf_loop_escalates_to_*/*/runs/* 2>/dev/null | head -1)
echo "Run dir: $RUN_DIR"
echo
echo "=== perf_repair attempts ==="
ls -la "$RUN_DIR/stage4_performance/perf_repair/"
echo
echo "=== Sonnet attempt_01 benchmark ==="
cat "$RUN_DIR/stage4_performance/perf_repair/attempt_01/benchmark.json" 2>/dev/null | python -m json.tool
echo
echo "=== Opus attempt_02 benchmark ==="
cat "$RUN_DIR/stage4_performance/perf_repair/attempt_02/benchmark.json" 2>/dev/null | python -m json.tool
echo
echo "=== Final report.json (perf section) ==="
python -c "import json; d=json.load(open('$RUN_DIR/report.json')); print(json.dumps(d.get('performance', {}), indent=2)); print(); print('=== stage_traces ==='); [print(t['stage_name'], t['attempts'], t['model_used']) for t in d['report']['stage_traces']]"
```

The path glob assumes `tmp_path`. If you set `artifact_root` to something else in the test, adjust accordingly.

---

## Cell 6 — Capture evidence for the milestone doc

```bash
!cd /content/Cuda-Engine && git rev-parse HEAD
!cd /content/Cuda-Engine && git log --oneline -1
```

Save the output. After the run, append a section to `docs/milestones/M3-evidence.md` (create if missing) with:
- Date and Colab session URL
- Commit SHA tested (should be `9d2d2d7` or later if the branch advanced)
- Test result (PASS / skip-reason / failure)
- The two `model_used` strings for the perf trace
- Sonnet final speedup, Opus final speedup
- Whether `below_target` was True or False

---

## If the test fails

Diagnose with `@superpowers:systematic-debugging`. Most likely causes:

1. **Sonnet's first kernel doesn't compile** → run dies in Stage 2 before reaching Stage 4. Check `stage2_codegen/escalated/` — codegen escalation may have fired instead, which is also fine but doesn't exercise the perf escalation we're testing. Re-run; if persistent, the prompt is too vague for softmax on this Colab build of nvcc. Tighten the prompt or use the simpler `vector_add` reference (which won't trigger perf escalation but at least isolates the Stage 2 path).

2. **Opus runs but writes to wrong attempt dir** → off-by-one in `attempt_offset` math. Check `_retry_loop` — `attempt = local_attempt + attempt_offset` should produce `attempt_02` when offset=1 and local_attempt=1. If it produces `attempt_03` instead, the offset is being passed as `retry_budget` (=1) but local starts at 1, so `1+1=2` is correct. Re-read the code.

3. **`model_used` only contains one model** → response.model field on the test fixtures isn't being threaded through. Check `_TracingLLMClient` (we deliberately do NOT stamp the model in production — the real Anthropic API returns `response.model` matching the request). If the API response has a different model field than the request, that's a real-API behavior we'd need to handle.

4. **`escalated to opus` not in notes** → the escalation block in `Stage4Performance.run()` didn't fire. Check whether `current_speedup >= target` was true after Sonnet — if Opus didn't run, escalation won't be in notes. This is the same as the "did not trigger" skip case.

---

## After Colab evidence lands

1. Append the evidence to `docs/milestones/M3-evidence.md`.
2. Update memory at `C:\Users\Shivnarain\.claude\projects\D--Cuda-Engine\memory\project_cuda_engine.md` — Task 4.3 complete with Colab confirmation, next task = 4.5.
3. Decide: merge `m3/perf-loop` to `main` now, or stack 4.5 on top of this branch first.
