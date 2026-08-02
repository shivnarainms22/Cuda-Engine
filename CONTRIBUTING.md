# Contributing

Thanks for looking. Issues and PRs are welcome, and so are questions in
[Discussions](https://github.com/shivnarainms22/Cuda-Engine/discussions).

## The one rule that matters

**Evidence before assertions.** This project's value is that its numbers survive
scrutiny, so a change is not done because it looks right — it is done when a
command was run and its output pasted. "Should work" and "looks good" are not
verification. Check the **exit code**, not whether the word `passed` appears in
the output; that exact mistake once hid a failing test for weeks.

If you add a performance claim, say what the baseline was and at which shape.
A speedup without a named baseline is not a result.

## Setup

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+. A GPU and CUDA toolchain are needed only for integration
tests and real synthesis — the unit suite is CPU-only by design, and most of the
engine can be developed and verified without a GPU or an API key.

## Before opening a PR

All three must exit 0:

```bash
pytest tests/unit
ruff check src tests evals
mypy src
```

Integration tests need CUDA and an API key, and cost money:

```bash
pytest tests/integration -v -m integration
```

## Tests

New behaviour needs tests covering the happy path, the error paths, and the edge
cases. Prefer real code over mocks; mock only unavoidable externals (network,
paid APIs). Name tests by behaviour, not implementation.

If you are testing that something is *detected*, include a **negative control** —
a case proving the test can fail. A green check that could never go red is not
evidence. See `tests/unit/test_torch_compat.py` for the pattern.

## Good first contributions

- **Kernels for the eval suite** (`evals/internal/`). Each fixture is a
  `prompt.txt`, `reference.py`, `shapes.yaml`, and `notes.md`. More coverage of
  the long tail is directly useful.
- **Architecture coverage.** Everything is verified on sm_80. Evidence from
  another architecture — even a failure report — is valuable.
- **Consumer integrations.** Emitting a Hugging Face `kernel-builder` layout or a
  FlashInfer-Bench Solution from `cuda-engine export` are both natural next steps.

## Scope

The engine targets elementwise ops, fusions, reductions/scans, fused GEMM
epilogues, and weight-only quantized kernels. Beating cuBLAS at bare GEMM and
beating FlashAttention at attention are explicitly out of scope — those are
solved by specialists, and pretending otherwise wastes everyone's time.
