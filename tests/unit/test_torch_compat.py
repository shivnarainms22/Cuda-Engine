"""Shape resolution and fake (meta) registration for generated kernels.

Every test here runs on CPU-only torch with no nvcc and no LLM call.
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from cuda_engine.models import KernelSpec, OptimizationPriority, PrecisionTolerance, TensorArg
from cuda_engine.torch_compat import (
    ShapeResolutionError,
    make_fake_impl,
    render_fake_module,
    resolve_output_shapes,
)


def _spec(
    *,
    inputs: list[tuple[str, str, tuple[str, ...]]],
    outputs: list[tuple[str, str, tuple[str, ...]]],
    name: str = "k",
) -> KernelSpec:
    return KernelSpec(
        name=name,
        target_arch="sm_80",
        inputs=[TensorArg(name=n, dtype=d, shape=s) for n, d, s in inputs],  # type: ignore[arg-type]
        outputs=[TensorArg(name=n, dtype=d, shape=s) for n, d, s in outputs],  # type: ignore[arg-type]
        precision_tolerance=PrecisionTolerance(),
        optimization_priority=OptimizationPriority.BALANCED,
    )


# --- Task 1: literal / symbol resolution, happy path ------------------------


def test_elementwise_symbol_resolves_from_input() -> None:
    spec = _spec(inputs=[("x", "fp32", ("N",))], outputs=[("y", "fp32", ("N",))])
    assert resolve_output_shapes(spec, [(1024,)]) == [(1024,)]


def test_literal_dim_in_output_is_passed_through() -> None:
    spec = _spec(inputs=[("x", "fp32", ("N",))], outputs=[("y", "fp32", ("4",))])
    assert resolve_output_shapes(spec, [(1024,)]) == [(4,)]


def test_literal_dim_in_input_is_not_treated_as_symbol() -> None:
    spec = _spec(inputs=[("x", "fp32", ("B", "128"))], outputs=[("y", "fp32", ("B",))])
    assert resolve_output_shapes(spec, [(8, 128)]) == [(8,)]


# --- Task 2: cross-argument symbol binding ---------------------------------


def test_matmul_binds_symbols_across_arguments() -> None:
    """The case stages/correctness.py::_concrete_shape cannot express."""
    spec = _spec(
        inputs=[("a", "fp16", ("M", "K")), ("b", "fp16", ("K", "N"))],
        outputs=[("c", "fp16", ("M", "N"))],
    )
    assert resolve_output_shapes(spec, [(64, 32), (32, 16)]) == [(64, 16)]


def test_symbol_bound_only_by_later_argument_resolves() -> None:
    spec = _spec(
        inputs=[("a", "fp16", ("M",)), ("b", "fp16", ("K", "N"))],
        outputs=[("c", "fp16", ("N", "M"))],
    )
    assert resolve_output_shapes(spec, [(5,), (7, 9)]) == [(9, 5)]


# --- Task 3: every failure is explicit -------------------------------------


def test_rank_mismatch_raises() -> None:
    spec = _spec(inputs=[("x", "fp32", ("B", "D"))], outputs=[("y", "fp32", ("B",))])
    with pytest.raises(ShapeResolutionError, match="rank"):
        resolve_output_shapes(spec, [(1024,)])


def test_literal_dim_mismatch_raises() -> None:
    spec = _spec(inputs=[("x", "fp32", ("B", "128"))], outputs=[("y", "fp32", ("B",))])
    with pytest.raises(ShapeResolutionError, match="128"):
        resolve_output_shapes(spec, [(8, 64)])


def test_conflicting_symbol_binding_raises() -> None:
    spec = _spec(
        inputs=[("a", "fp16", ("M", "K")), ("b", "fp16", ("K", "N"))],
        outputs=[("c", "fp16", ("M", "N"))],
    )
    with pytest.raises(ShapeResolutionError, match="K"):
        resolve_output_shapes(spec, [(64, 32), (99, 16)])


def test_unbound_output_symbol_raises() -> None:
    spec = _spec(inputs=[("x", "fp32", ("N",))], outputs=[("y", "fp32", ("Z",))])
    with pytest.raises(ShapeResolutionError, match="Z"):
        resolve_output_shapes(spec, [(1024,)])


def test_too_few_arguments_raises() -> None:
    spec = _spec(
        inputs=[("a", "fp16", ("M", "K")), ("b", "fp16", ("K", "N"))],
        outputs=[("c", "fp16", ("M", "N"))],
    )
    with pytest.raises(ShapeResolutionError, match="argument"):
        resolve_output_shapes(spec, [(64, 32)])


def test_no_silent_fallback_when_spec_has_no_inputs() -> None:
    spec = _spec(inputs=[], outputs=[("y", "fp32", ("N",))])
    with pytest.raises(ShapeResolutionError):
        resolve_output_shapes(spec, [])


# --- Task 4: scalar / 0-D and multi-output ---------------------------------


def test_scalar_input_and_scalar_output() -> None:
    spec = _spec(
        inputs=[("x", "fp32", ("N",)), ("alpha", "fp32", ())],
        outputs=[("total", "fp32", ())],
    )
    assert resolve_output_shapes(spec, [(32,), ()]) == [()]


def test_reduction_output_drops_a_dimension() -> None:
    spec = _spec(inputs=[("x", "fp16", ("B", "D"))], outputs=[("y", "fp16", ("B",))])
    assert resolve_output_shapes(spec, [(12, 256)]) == [(12,)]


def test_multiple_outputs_resolve_independently() -> None:
    spec = _spec(
        inputs=[("x", "fp32", ("B", "D"))],
        outputs=[("vals", "fp32", ("B",)), ("idx", "int64", ("B", "D"))],
    )
    assert resolve_output_shapes(spec, [(4, 9)]) == [(4,), (4, 9)]


# --- Task 5-7: the fake implementation under torch.compile ------------------

torch = pytest.importorskip("torch")

_NS_COUNTER = itertools.count()


@pytest.fixture
def make_op() -> Any:
    """Define a throwaway torch.library op with a CPU impl, optionally faked.

    A fresh namespace per call: a namespace can only be defined once per process.
    """
    held: list[Any] = []

    def _make(spec: KernelSpec, schema: str, cpu_impl: Any, *, with_fake: bool) -> Any:
        ns = f"cuda_engine_test_{next(_NS_COUNTER)}"
        lib = torch.library.Library(ns, "DEF")
        held.append(lib)
        lib.define(f"forward{schema}")
        lib.impl("forward", cpu_impl, "CPU")
        if with_fake:
            torch.library.register_fake(f"{ns}::forward", make_fake_impl(spec))
        return getattr(torch.ops, ns).forward

    yield _make
    for lib in held:
        lib._destroy()


def test_fake_impl_returns_single_tensor_with_resolved_shape_and_dtype() -> None:
    spec = _spec(inputs=[("x", "fp32", ("B", "D"))], outputs=[("y", "fp16", ("B",))])
    fake = make_fake_impl(spec)
    with torch._subclasses.fake_tensor.FakeTensorMode():
        out = fake(torch.empty(6, 10))
    assert tuple(out.shape) == (6,)
    assert out.dtype == torch.float16


def test_fake_impl_returns_tuple_for_multiple_outputs() -> None:
    spec = _spec(
        inputs=[("x", "fp32", ("B", "D"))],
        outputs=[("vals", "fp32", ("B",)), ("idx", "int64", ("B", "D"))],
    )
    fake = make_fake_impl(spec)
    with torch._subclasses.fake_tensor.FakeTensorMode():
        vals, idx = fake(torch.empty(3, 5))
    assert tuple(vals.shape) == (3,)
    assert tuple(idx.shape) == (3, 5)
    assert idx.dtype == torch.int64


def test_fake_impl_inherits_device_from_first_input() -> None:
    spec = _spec(inputs=[("x", "fp32", ("N",))], outputs=[("y", "fp32", ("N",))])
    fake = make_fake_impl(spec)
    with torch._subclasses.fake_tensor.FakeTensorMode():
        out = fake(torch.empty(4))
    assert out.device.type == "cpu"


def test_fake_impl_propagates_resolution_errors() -> None:
    spec = _spec(inputs=[("x", "fp32", ("B", "D"))], outputs=[("y", "fp32", ("Z",))])
    fake = make_fake_impl(spec)
    with pytest.raises(ShapeResolutionError, match="Z"), torch._subclasses.fake_tensor.FakeTensorMode():
        fake(torch.empty(2, 2))


# -- Task 6: acceptance -- fullgraph tracing, with a negative control --------


def _elementwise_case() -> tuple[KernelSpec, str, Any]:
    spec = _spec(inputs=[("x", "fp32", ("N",))], outputs=[("y", "fp32", ("N",))])
    return spec, "(Tensor x) -> Tensor", lambda x: x.clone()


def test_negative_control_without_fake_impl_fullgraph_tracing_fails(make_op: Any) -> None:
    """Proves the acceptance test below can fail -- otherwise it proves nothing."""
    spec, schema, cpu_impl = _elementwise_case()
    op = make_op(spec, schema, cpu_impl, with_fake=False)

    def fn(t: Any) -> Any:
        return op(t) + 1

    torch._dynamo.reset()
    with pytest.raises(Exception):  # noqa: B017 - any Dynamo failure is the signal
        torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(8))


def test_fullgraph_tracing_succeeds_with_generated_fake(make_op: Any) -> None:
    """THE acceptance criterion: the op composes inside a torch.compile graph."""
    spec, schema, cpu_impl = _elementwise_case()
    op = make_op(spec, schema, cpu_impl, with_fake=True)

    def fn(t: Any) -> Any:
        return op(t) + 1

    torch._dynamo.reset()
    result = torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(8))
    assert tuple(result.shape) == (8,)


def test_dynamic_shapes_do_not_recompile(make_op: Any) -> None:
    """A SymInt coerced to int would specialize and force a second compile."""
    spec, schema, cpu_impl = _elementwise_case()
    op = make_op(spec, schema, cpu_impl, with_fake=True)
    compiles = 0

    def counting_backend(gm: Any, example_inputs: Any) -> Any:
        nonlocal compiles
        compiles += 1
        return gm.forward

    def fn(t: Any) -> Any:
        return op(t) + 1

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend=counting_backend, dynamic=True, fullgraph=True)
    compiled(torch.randn(8))
    compiled(torch.randn(16))
    assert compiles == 1


# --- Task 8: standalone rendered module ------------------------------------


def _render_and_exec(spec: KernelSpec) -> dict[str, Any]:
    source = render_fake_module(spec)
    compile(source, "<generated>", "exec")  # must be syntactically valid
    namespace: dict[str, Any] = {}
    exec(source, namespace)  # exercising the generated source is the point
    return namespace


def test_rendered_module_is_standalone_and_resolves_shapes() -> None:
    spec = _spec(
        inputs=[("a", "fp16", ("M", "K")), ("b", "fp16", ("K", "N"))],
        outputs=[("c", "fp16", ("M", "N"))],
    )
    source = render_fake_module(spec)
    # Standalone: the exported package must not depend on the generator. The op
    # qualname "cuda_engine::forward" legitimately appears, so check imports only.
    assert "import cuda_engine" not in source
    assert "from cuda_engine" not in source
    namespace = _render_and_exec(spec)
    assert namespace["resolve_output_shapes"]([(64, 32), (32, 16)]) == [(64, 16)]


def test_rendered_module_matches_in_process_resolution() -> None:
    spec = _spec(
        inputs=[("x", "fp32", ("B", "D"))],
        outputs=[("vals", "fp32", ("B",)), ("idx", "int64", ("B", "D"))],
    )
    namespace = _render_and_exec(spec)
    assert namespace["resolve_output_shapes"]([(4, 9)]) == resolve_output_shapes(spec, [(4, 9)])


def test_rendered_module_raises_the_same_errors() -> None:
    spec = _spec(inputs=[("x", "fp32", ("N",))], outputs=[("y", "fp32", ("Z",))])
    namespace = _render_and_exec(spec)
    with pytest.raises(ValueError, match="Z"):
        namespace["resolve_output_shapes"]([(8,)])


def test_rendered_module_exposes_a_registrar() -> None:
    spec = _elementwise_case()[0]
    namespace = _render_and_exec(spec)
    assert callable(namespace["register"])
    assert callable(namespace["fake_forward"])


def test_rendered_module_annotations_resolve() -> None:
    """Embedded source must carry its own imports, not rely on lazy annotations."""
    import typing

    spec = _spec(inputs=[("x", "fp32", ("N",))], outputs=[("y", "fp32", ("N",))])
    namespace = _render_and_exec(spec)
    typing.get_type_hints(namespace["_resolve_shapes"], globalns=namespace)
    typing.get_type_hints(namespace["resolve_output_shapes"], globalns=namespace)


def test_rendered_module_register_enables_fullgraph_tracing() -> None:
    """End-to-end through the *generated source*, not the in-process impl."""
    spec, schema, cpu_impl = _elementwise_case()
    ns = f"cuda_engine_test_{next(_NS_COUNTER)}"
    lib = torch.library.Library(ns, "DEF")
    try:
        lib.define(f"forward{schema}")
        lib.impl("forward", cpu_impl, "CPU")
        namespace = _render_and_exec(spec)
        namespace["register"](f"{ns}::forward")
        op = getattr(torch.ops, ns).forward

        def fn(t: Any) -> Any:
            return op(t) + 1

        torch._dynamo.reset()
        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(8))
        assert tuple(result.shape) == (8,)
    finally:
        lib._destroy()
