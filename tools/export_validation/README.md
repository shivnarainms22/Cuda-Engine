# Export validation

Proves the **export → install → use** loop works on real hardware.

The unit suite (`tests/unit/test_export.py`, `tests/unit/test_torch_compat.py`) runs
on CPU and proves the package *files* are generated correctly. It cannot prove the
package builds, loads, computes the right answer, and traces under `torch.compile`
on a GPU. That is what this does.

## Run it

```bash
python tools/export_validation/validate_export.py --run-id <run_id> [--runs-root DIR]
```

Consumes an existing run and makes **no LLM calls** — it costs nothing beyond the
GPU. Exit code 0 is the verdict; do not read the log for it.

For Colab, use [`examples/export_validation.ipynb`](../../examples/export_validation.ipynb).

## What it does

1. `cuda-engine export` the run into a temp directory, then **confirm the package
   exists** — a CLI invoked the wrong way can exit 0 having done nothing.
2. `pip install` it.
3. Run `standalone_check.py` in a **fresh subprocess whose working directory is not
   the repo**, so `import cuda_engine` cannot resolve locally. If the exported
   package secretly needed the generator, this is where it shows.

## What `standalone_check.py` asserts

| Check | Why it matters |
|---|---|
| imports without `cuda_engine` | the package must stand alone |
| kernel builds and runs | the JIT `cpp_extension.load` / `.so` path has otherwise never executed |
| matches the reference | the exported kernel must compute what was verified |
| **control:** comparison rejects a wrong answer | without this the check above proves nothing |
| `torch.compile(fullgraph=True)` traces it | the entire reason the fake/meta impl exists |
| compiled result matches eager | tracing succeeding is not the same as being correct |

The control is not decoration. The WMMA guidance harness earned its result by
having a negative control; a green check that cannot go red is not evidence.

## Known limitation

The op is registered under the fixed `cuda_engine::forward` namespace, so **two
exported packages cannot be imported into the same process** — the second
`TORCH_LIBRARY` registration collides. Fine for validation and for using one
generated kernel; it needs a per-package namespace before anyone ships several.
