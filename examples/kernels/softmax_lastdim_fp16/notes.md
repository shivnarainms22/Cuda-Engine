# softmax_lastdim_fp16

Softmax over the last dim of a 2D fp16 tensor. Two-pass classic with the max-subtraction stabilization trick.

**Why it's a good worked example:**
- Hits the "two-pass reduction → elementwise" pattern (max-find, then exp-sum, then divide).
- Hot-path kernel for transformer architectures, so the performance bar is realistic.
- Demonstrates fp32 accumulation alongside fp16 storage.

**Expected speedup band on A100:** ~0.9–1.2× torch.compile. torch.compile's softmax is highly tuned, so this is a "match or modestly beat" target. Genuinely beating it usually requires warp-level reductions + one-shot fusion of all three passes via shared memory.

## How to populate run_dir/

```bash
cuda-engine synthesize \
    --prompt-file examples/kernels/softmax_lastdim_fp16/prompt.txt \
    --reference examples/kernels/softmax_lastdim_fp16/reference.py \
    --target sm_80 \
    --out examples/kernels/softmax_lastdim_fp16/run_dir
```
