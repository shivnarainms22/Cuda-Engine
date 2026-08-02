# Kernel Export — Design

**Status:** Approved (2026-08-01)
**Increment:** v2.2 — deployability track, item 2 of 4. Branch `v2.2/torch-compat`.
**Depends on:** `cuda_engine.torch_compat` (item 1).
**Hardware:** none required to *generate* an export. Consuming one needs a GPU.

## 1. Problem

A successful `synthesize()` ends with a `.cu` file inside
`~/.cache/cuda_engine/runs/<run_id>/stage5_polish/final/`. That is a *result*, not
a *dependency*. To actually use it a person must find the run dir, hand-write a
`cpp_extension.load` call, hand-register a fake impl, and re-derive from the JSON
what was verified and at which shapes. Nobody does this, so nobody depends on the
engine's output.

## 2. Goal

`cuda-engine export <run_id> --out <dir>` produces a self-contained, importable
Python package for one verified kernel, carrying its own verification evidence.

**Done =** the exported directory can be `pip install`ed, `import`ed, called, and
traced by `torch.compile`; and it states exactly what was verified and what was
not.

## 3. Locked decisions

### D1 — The core is a pure function; the filesystem is a thin shell

`build_export(...) -> dict[str, str]` maps relative path → file content. The CLI
writes that mapping to disk. All layout, rendering and integrity logic is
therefore unit-testable with no filesystem and no GPU.

### D2 — Refuse to export an unverified kernel

Export is a *trust* artifact. If the run's correctness gate did not pass, export
fails with a clear message. Performance below target does **not** block (soft
gate) but is recorded verbatim.

`--force` overrides, and when used, stamps `VERIFICATION.md` with a prominent
UNVERIFIED banner. There is no silent path to an unmarked unverified package.

### D3 — Ship source, JIT-build on first import

The package ships `kernel.cu` and builds via `torch.utils.cpp_extension.load`,
cached by torch. Rejected: shipping only the prebuilt `.so`, which silently binds
the package to one exact torch/CUDA/driver/arch combination — the failure mode is
a confusing runtime symbol error rather than an honest rebuild. The prebuilt `.so`
*is* copied in when present and used when it loads, with source build as fallback.

Rejected for now: the Hugging Face `kernel-builder` multi-arch Nix matrix. It is
the right eventual target for binary distribution, but it is a build-system
dependency and a separate increment. The layout here does not preclude it.

### D4 — Prefer the polished kernel, and say which one shipped

`stage5_polish/final/kernel.cu` exists only when the annotated kernel recompiled
*and* re-passed correctness. Prefer it; fall back to the accepted artifact path.
The manifest records which was used — never leave the consumer guessing whether
they got the annotated or the raw kernel.

### D5 — `VERIFICATION.md` states the negative space

The evidence file records what passed *and* what was not checked: runtime
verification is sm_80-only, tolerances used, the shapes actually tested, the
torch.compile baseline mode the speedup was measured against, and whether
correctness was confirmed at the benchmark shape. A verification document that
only lists successes is marketing, not evidence.

## 4. Layout produced

```
<out>/
  pyproject.toml          # pip-installable
  README.md               # usage
  VERIFICATION.md         # evidence, including negative space
  <pkg>/
    __init__.py           # load .so or JIT-build, register fake, expose forward()
    fake_impl.py          # render_fake_module(spec)
    kernel.cu
    spec.json
    manifest.json         # provenance: run_id, engine version, source kernel, date
    kernel.so             # optional, when the run produced one
```

`<pkg>` is the spec name sanitized to a Python identifier, prefixed `ce_` to avoid
colliding with a module the consumer already has.

## 5. Scope

**In:** `src/cuda_engine/export.py` (`build_export`, `ExportError`, `write_export`)
and a `cuda-engine export` CLI subcommand.

**Out:** wheel building, HF Hub upload, FlashInfer-Bench Solution emission,
multi-arch binaries. All are follow-ons this layout supports.

## 6. Verification (no GPU, no credits)

Unit tests drive `build_export` from an `InMemoryStore` populated to look like a
real run, asserting: refusal on failed correctness; `--force` banner; polished-vs-
fallback kernel selection; every file present; `__init__.py` and `fake_impl.py`
are syntactically valid (`compile()`); `pyproject.toml` parses (`tomllib`); the
rendered fake resolves the spec's shapes. The JIT-build path itself needs a GPU
and is left to integration tests.
