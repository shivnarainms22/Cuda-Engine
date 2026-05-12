# rmsnorm_silu_fp16

Fused RMSNorm + SiLU over the last dim. Single pass over the input, with the per-row mean-of-squares computed in fp32 and the output written back as fp16.

**Why it's a good worked example:**
- Demonstrates a real fusion that's worth doing (avoids the second pass over the tensor that an unfused implementation would need).
- Hits both reduction (mean over last dim) and elementwise (divide + silu) patterns.
- fp32 accumulator path is the kind of numerical-stability detail Stage 1 should pull out of the prompt.

**Expected speedup band on A100:** ~1.0–1.5× torch.compile. torch.compile fuses these patterns reasonably well, so beating it requires careful warp-level reductions + `__half2` vectorization.

## How to populate run_dir/

Run on a Colab A100 with cuda-engine installed and `ANTHROPIC_API_KEY` set:

```bash
cuda-engine synthesize \
    --prompt-file examples/kernels/rmsnorm_silu_fp16/prompt.txt \
    --reference examples/kernels/rmsnorm_silu_fp16/reference.py \
    --target sm_80 \
    --out examples/kernels/rmsnorm_silu_fp16/run_dir
```

Then commit the resulting `run_dir/` (or a sanitized subset — see `docs/privacy.md`).
