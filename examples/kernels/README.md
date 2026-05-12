# Worked examples

Each subdirectory here is a self-contained worked example: a real prompt + reference function that you can run through cuda-engine to produce an annotated kernel and a synthesis report.

The directory layout matches `evals/internal/<name>/`:

```
<example_name>/
  prompt.txt          # natural-language description
  reference.py        # top-level reference() or REFERENCE
  shapes.yaml         # input shapes used for correctness testing
  notes.md            # what makes this example interesting + how to run it
  run_dir/            # populated after running cuda-engine synthesize (gitignored by default)
    report.json
    stage1_interview/...
    stage2_codegen/...
    stage5_polish/final/kernel.cu      # the final annotated kernel
    ...
```

## Running an example

```bash
export ANTHROPIC_API_KEY=sk-ant-...

cuda-engine synthesize \
    --prompt-file examples/kernels/rmsnorm_silu_fp16/prompt.txt \
    --reference examples/kernels/rmsnorm_silu_fp16/reference.py \
    --target sm_80 \
    --out examples/kernels/rmsnorm_silu_fp16/run_dir
```

Each example's `notes.md` includes the exact command.

## The current examples

- **`rmsnorm_silu_fp16/`** — Fused RMSNorm + SiLU over the last dim. Exercises reduction + elementwise fusion.
- **`softmax_lastdim_fp16/`** — Numerically stable softmax with the max-subtraction trick. Classic two-pass reduction → elementwise pattern.
- **`topk_fp32/`** — Per-row top-k with values + indices. Multi-output kernel; algorithmic variety.

## Why these three

They cover the three different shapes a v1-scope CUDA kernel can take:

| Shape | Example | Why it matters |
|---|---|---|
| Pure elementwise / lightweight fusion | rmsnorm_silu_fp16 | The bread-and-butter case for v1; combines reduction + activation. |
| Multi-pass reduction with numerical-stability dance | softmax_lastdim_fp16 | Hot-path for transformer inference; competitive vs torch.compile is hard. |
| Multi-output kernel with algorithmic choice | topk_fp32 | Demonstrates non-trivial output structure; relevant for sampling/beam-search workloads. |

If cuda-engine produces clean kernels for these three, it covers the v1 design space.

## Adding your own

To add a new worked example, mirror one of the existing directories:

1. Create `examples/kernels/<your_name>/`.
2. Write a tight `prompt.txt` (1–3 sentences) describing the kernel.
3. Write `reference.py` with a top-level `reference(...)` function.
4. Write `shapes.yaml` with at least 3 shapes (cover edge cases).
5. Write `notes.md` explaining what the example demonstrates + the run command.
6. Run `cuda-engine synthesize` to populate `run_dir/`.
7. (Optional) Commit a sanitized `run_dir/` — see `docs/privacy.md` for what to redact.
