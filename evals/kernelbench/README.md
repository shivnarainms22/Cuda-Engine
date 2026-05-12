# KernelBench eval suite

External eval suite based on [KernelBench](https://github.com/ScalingIntelligence/KernelBench) (MIT license).

`evals/kernelbench/filtered/` holds a hand-curated subset of KernelBench files translated into the `prompt.txt` + `reference.py` + `shapes.yaml` + `notes.md` layout that `cuda-engine eval --suite kernelbench` expects. We do **not** auto-translate KernelBench into our format — the source classes carry contextual assumptions (input shapes, dtypes, expected error tolerances) that don't survive a mechanical conversion safely.

Instead, this directory ships a **screening tool** (`evals/kernelbench/filter.py`) that classifies every KernelBench file by whether it falls within cuda-engine v1's scope, plus a workflow for adding new entries to `filtered/` by hand.

## v1 scope

In scope:
- Elementwise ops (relu, gelu, silu, sigmoid, tanh, etc.).
- Simple fused elementwise patterns (bias+gelu, swiglu, geglu, etc.).
- Normalizations without affine parameters (rms_norm, layer_norm).
- Reductions and scans (sum, mean, max, argmax, top-k, prefix-sum).
- Masked / dropout / clamp / type conversion variants.

Out of scope (v1):
- Matmul, batched matmul, einsum.
- Attention (`scaled_dot_product_attention`, multi-head attention).
- Convolutions (any rank, including transposed).
- Sequence models (RNN, LSTM, GRU).
- FFT / spectral ops.
- Anything requiring autograd or backward-pass kernels.

## Workflow

### 1. Clone KernelBench locally

```bash
git clone --depth 1 https://github.com/ScalingIntelligence/KernelBench.git \
    ~/.cache/cuda_engine/kernelbench
```

### 2. Run the filter

```bash
python -m evals.kernelbench.filter \
    --kernelbench-root ~/.cache/cuda_engine/kernelbench \
    --out evals/kernelbench/candidates
```

This produces:
- `candidates.json` — full per-file classification record.
- `candidates.md` — human-readable summary grouped by verdict.

### 3. Hand-translate `in_scope` entries you want to evaluate

For each KernelBench `level1/<NN>_<op>.py` you want to include, create `evals/kernelbench/filtered/<our_name>/` with:

- `prompt.txt` — short natural-language description of the kernel.
- `reference.py` — defines a top-level `reference(...)` function with the same signature you'd expect cuda-engine to handle. Translate KernelBench's `Model.forward(...)` here, dropping `self.` references for tensor-only inputs.
- `shapes.yaml` — at least 3 input shapes covering boundaries (0, 1, small, large, off-by-one).
- `notes.md` — link back to the KernelBench source file + any context (expected speedup band, edge cases).

### 4. Run the eval

```bash
cuda-engine eval \
    --suite kernelbench \
    --out evals/results/kernelbench-$(date +%Y-%m-%d) \
    --resume
```

`--suite kernelbench` maps to `evals/kernelbench/filtered/` automatically.

## Why not auto-translate?

A KernelBench Model's `forward` looks innocent but often depends on hyperparameters set in `__init__` (shape, dtype, normalization params). Auto-extracting just the forward body can:
- Produce references with the wrong signature.
- Hide affine parameters that change the kernel's behavior.
- Inherit shape assumptions that don't survive shape-grid testing.

The 30-kernel internal suite was hand-curated for exactly this reason — small enough to verify, well-shaped for the v1 pipeline. The KernelBench subset extends coverage but trades automation for safety.

## License

KernelBench is licensed under MIT. The contents of `evals/kernelbench/filtered/` are **derivative works** of the original KernelBench source files. Each translated kernel's `notes.md` should cite the original `level<N>/<NN>_<name>.py` path. The KernelBench LICENSE applies to those derived fixtures.
