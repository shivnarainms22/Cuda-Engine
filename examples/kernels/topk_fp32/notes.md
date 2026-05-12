# topk_fp32

Per-row top-k along the last dim, returning both values and indices, sorted descending. k is a runtime scalar parameter.

**Why it's a good worked example:**
- Different shape from elementwise / reduction: it returns a (values, indices) tuple, exercising multi-output kernels.
- Algorithmic variety: the kernel can choose between bitonic sort, partial selection, or warp-level reduction strategies depending on k and row length.
- Real-world: top-k is in every beam search, every recommender, every sampling loop.

**Expected speedup band on A100:** highly k-dependent. For small k (≤32) and small rows, a hand-rolled CUDA implementation typically matches `torch.topk`. For larger rows, beating Torch becomes feasible via warp-level reductions.

## How to populate run_dir/

```bash
cuda-engine synthesize \
    --prompt-file examples/kernels/topk_fp32/prompt.txt \
    --reference examples/kernels/topk_fp32/reference.py \
    --target sm_80 \
    --out examples/kernels/topk_fp32/run_dir
```
