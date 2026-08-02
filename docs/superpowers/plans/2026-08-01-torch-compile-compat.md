# torch.compile Compatibility — Implementation Plan

Spec: `docs/superpowers/specs/2026-08-01-torch-compile-compat-design.md`
Branch: `v2.2/torch-compat` off `main`… **correction:** off current `v2.1/quant`
is wrong (unrelated in-flight work). Branch off `main`.

Files touched:
- `src/cuda_engine/torch_compat.py` (new)
- `tests/unit/test_torch_compat.py` (new)

Commands (run after every task):
```
.venv/Scripts/python -m pytest tests/unit -q
.venv/Scripts/python -m ruff check src tests evals
.venv/Scripts/python -m mypy src
```
Expected: exit code 0 for each. **Verify by exit code, not by grepping "passed".**

---

## Task 1 — `ShapeResolutionError` + literal/symbol resolution for the happy path

*Failing test:* `resolve_output_shapes` for an elementwise spec
(`in: x:(N,) fp32` → `out: y:(N,) fp32`) given `[(1024,)]` returns `[(1024,)]`.

*Implement:* symbol table build + output resolution (spec §D2). Literal dims via
`int(dim)`; symbols recorded into a shared dict.

## Task 2 — cross-argument symbol binding

*Failing test:* rank-2 matmul spec `a:(M,K)`, `b:(K,N)` → `out:(M,N)` with inputs
`[(64,32),(32,16)]` resolves to `[(64,16)]`. This is the case
`stages/correctness.py:_concrete_shape` cannot do.

*Implement:* ensure the table is shared across args, not per-arg.

## Task 3 — every D3 error condition raises

*Failing tests* (one per row of the spec §D3 table): rank mismatch; literal
mismatch; conflicting symbol binding; unbound output symbol; too few args.

*Implement:* the five guards. No fallbacks.

## Task 4 — scalar / 0-D and multi-output

*Failing test:* spec with `shape: []` input and a reduction spec
`x:(B,D)` → `out:(B,)`; and a two-output spec resolving both.

*Implement:* whatever the tests expose (empty tuple handling).

## Task 5 — `make_fake_impl` returns a working fake

*Failing test:* under `FakeTensorMode`, calling `make_fake_impl(spec)(*fake_inputs)`
returns tensors with the right shape, dtype, and device. Single output returns a
tensor; multi-output returns a tuple.

*Implement:* build outputs with `new_empty` on the first input tensor, dtype from
`spec.outputs[j].dtype` via the existing `_torch_dtype_name` mapping.

## Task 6 — ACCEPTANCE: `torch.compile(fullgraph=True)` traces the op

*Failing test:* define a real op with
`torch.library.Library("cuda_engine_test_ns", "DEF")`, register the generated fake
via `torch.library.register_fake`, then `torch.compile(fn, fullgraph=True)` over a
function that calls it. Must not raise (a graph break under `fullgraph=True` raises).

*Implement:* only fixes needed to make it pass. This is the criterion that the
whole increment exists for.

## Task 7 — SymInt safety under `dynamic=True`

*Failing test:* compile with `dynamic=True` across two different sizes and assert
only one compilation occurs (no recompile ⇒ no specialization), and that the
resolver never called `int()` on a runtime dim.

*Implement:* remove any `int()` coercion of runtime sizes; literal comparison via `==`.

## Task 8 — `render_fake_module` emits standalone source

*Failing test:* `render_fake_module(spec)` output (a) compiles via `compile()`,
(b) when `exec`'d exposes a callable that resolves the same shapes as
`make_fake_impl`, (c) embeds no import of `cuda_engine` (must be standalone for the
export package).

*Implement:* render the spec's shape data as literals into a template.

## Task 9 — docs + commit

CHANGELOG `[Unreleased]` entry. Final full verification run.
