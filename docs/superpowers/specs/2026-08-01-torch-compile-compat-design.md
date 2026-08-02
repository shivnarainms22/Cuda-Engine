# torch.compile Compatibility (fake/meta registration) — Design

**Status:** Approved (2026-08-01)
**Increment:** v2.2 — deployability track, item 1 of 4. Branch `v2.2/torch-compat`.
**Hardware:** none required. Fully verifiable on CPU-only torch. Zero API credits.

## 1. Problem

`prompts/codegen.md:9-10` requires every generated kernel to register
`TORCH_LIBRARY(cuda_engine, m)` + `TORCH_LIBRARY_IMPL(cuda_engine, CUDA, m)`.
There is **no Meta/fake implementation**. Consequences:

- `torch.compile` cannot trace `torch.ops.cuda_engine.forward`. It emits a graph
  break, splitting the compiled region, killing fusion across the boundary and
  disqualifying the region from CUDA graphs.
- `torch.export` and AOTInductor fail outright — FakeTensor propagation has no
  shape rule for the op.

So the artifact this engine produces cannot be used inside the very system it
benchmarks against. A kernel measured at 1.25× vs torch.compile cannot be *put
into* a torch.compile graph. That is the gap between "a result" and "a
dependency", and it is the precondition for the export/distribution work that
follows.

## 2. Goal

Every synthesized kernel ships with a correct, mechanically-derived fake
(meta) implementation, and the engine records hard evidence that the op traces
cleanly under `torch.compile(fullgraph=True)`.

**Done =** given a `KernelSpec`, the engine can (a) resolve output shapes from
input shapes, (b) register a fake impl for `cuda_engine::forward`, (c) emit that
fake as standalone Python source for shipping alongside the kernel, and (d)
prove `fullgraph=True` tracing succeeds — all unit-tested on CPU with no GPU and
no LLM call.

## 3. Locked decisions

### D1 — Python `torch.library.register_fake`, not a C++ Meta impl

The fake is **generated deterministically from the `KernelSpec`**, in Python.
The LLM is not involved.

Rejected: teaching `codegen.md` to emit `TORCH_LIBRARY_IMPL(cuda_engine, Meta, m)`.
That puts shape logic in LLM-written C++ — a new failure mode inside the repair
loop, costing tokens and retry budget, for information the engine *already has
exactly* in the frozen spec. v2.0 Rung 4 is the precedent: adding prompt guidance
to fix a codegen gap regressed a validated win. Do not put derivable facts in the
prompt.

Python `register_fake` is also the officially documented path for custom ops, and
it is testable without nvcc.

### D2 — Symbol binding is global across arguments, from real shapes

`stages/correctness.py:95` (`_concrete_shape`) binds symbolic dims *positionally
per-arg* against a fallback tuple, with no cross-argument binding — the known
limitation that forced v2.0 GEMM to square-only shapes.

The fake impl does **not** inherit that limitation. At trace time the actual input
shapes are given, so binding is exact:

1. For each spec input *i*, zip `spec.inputs[i].shape` against `args[i].shape`.
2. A dim that parses as an integer is a literal — check it matches, else raise.
3. Otherwise it is a symbol — record `symbol -> size` in one table shared across
   *all* arguments.
4. Resolve each `spec.outputs[j].shape` against that table.

This is strictly more capable than `_make_inputs` and is the correct semantics.

### D3 — Every failure is explicit; never guess a shape

A silently-wrong fake produces silently-wrong downstream shapes — precisely the
class of defect this project exists to catch (cf. the fake GEMM wins that were
correct at 1024² and garbage at 4096²). So:

| Condition | Behaviour |
|---|---|
| Rank of actual arg ≠ rank in spec | raise `ShapeResolutionError` |
| Literal dim in spec ≠ actual size | raise `ShapeResolutionError` |
| Same symbol bound to two sizes | raise `ShapeResolutionError` |
| Output symbol never bound by any input | raise `ShapeResolutionError` |
| Fewer args than spec inputs | raise `ShapeResolutionError` |

No fallbacks, no defaults. An unresolvable spec means "this kernel cannot have a
derived fake" and must be reported as such, not papered over.

### D4 — SymInt-safe: never coerce a dim to `int`

Under `torch.compile(dynamic=True)` sizes are `SymInt`. The resolver propagates
whatever size object it receives into `new_empty`, and only calls `int()` on dims
that came from the *spec* string. Dynamic shapes therefore work for free, and a
`SymInt` never gets specialized into a constant by our code.

Equality checks against literals use `==`, which is SymInt-safe (guards, not
specialization).

### D5 — Soft gate, recorded

torch.compile compatibility is **recorded, not run-failing**, mirroring the
existing hard-gate (correctness) / soft-gate (performance) split. A kernel that
is correct and fast is still valuable if its fake cannot be derived (e.g. a
data-dependent output shape). The result lands in the report so the user knows
what they have. Config knob `require_torch_compile_compat: bool = False` lets a
consumer harden it.

## 4. Scope

**In:** a single new module `src/cuda_engine/torch_compat.py` exposing
- `resolve_output_shapes(spec, input_shapes) -> list[tuple[Any, ...]]`
- `make_fake_impl(spec) -> Callable[..., Any]`
- `render_fake_module(spec) -> str` — standalone Python source for the export package
- `ShapeResolutionError`

**Out (later increments):** the `cuda-engine export` command that consumes
`render_fake_module` (item 2); orchestrator/report wiring; README repositioning.
Keeping this increment to pure, dependency-free logic is what makes it 100%
CPU-testable.

## 5. Verification strategy (no GPU, no credits)

`torch 2.11.0+cpu` is installed locally and exposes `torch.library.register_fake`.
Tests define a throwaway op with `torch.library.Library("cuda_engine_test", "DEF")`,
attach the generated fake, and assert:

- output shapes/dtypes under `FakeTensorMode` for representative specs
  (elementwise, reduction, rank-2 matmul with cross-arg symbols, scalar/0-D,
  multi-output);
- every D3 error condition raises;
- `torch.compile(fullgraph=True)` traces a function calling the op without a
  graph break — the actual acceptance criterion;
- dynamic shapes (`dynamic=True`) do not specialize.

The real CUDA op is never needed: `fullgraph` tracing exercises the meta path only.
