"""Derive a fake (meta) implementation for a generated kernel from its KernelSpec.

Generated kernels register ``TORCH_LIBRARY(cuda_engine, m)`` plus a CUDA impl, but
a CUDA impl alone is opaque to Dynamo: ``torch.compile`` graph-breaks on the op and
``torch.export``/AOTInductor fail because FakeTensor propagation has no shape rule.

The shape rule is already known exactly -- it is the frozen ``KernelSpec``. This
module derives it mechanically. The LLM is not involved, so there is no new failure
mode in the repair loop and the whole path is testable on CPU-only torch.

See ``docs/superpowers/specs/2026-08-01-torch-compile-compat-design.md``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any

from cuda_engine.models import KernelSpec
from cuda_engine.models.spec import TORCH_DTYPE_NAMES

__all__ = [
    "ShapeResolutionError",
    "make_fake_impl",
    "render_fake_module",
    "resolve_output_shapes",
]

#: Shape declarations as plain data: ``(arg_name, symbolic_dims)``. Used instead of
#: ``TensorArg`` by the pure core so the same code can run inside a rendered module
#: that must not import this package.
ShapeDecl = tuple[str, tuple[str, ...]]


class ShapeResolutionError(ValueError):
    """Output shapes cannot be derived from the given input shapes.

    Always raised rather than guessing. A silently-wrong fake produces
    silently-wrong downstream shapes, which is the defect class this project
    exists to catch.
    """


def _literal_dim(dim: str) -> int | None:
    """Return the integer value of a declared dim, or None if it is a symbol."""
    try:
        return int(dim)
    except ValueError:
        return None


def _resolve_shapes(
    kernel_name: str,
    input_decls: Sequence[ShapeDecl],
    output_decls: Sequence[ShapeDecl],
    input_shapes: Sequence[Sequence[Any]],
) -> list[tuple[Any, ...]]:
    """Bind symbols from concrete input shapes, then resolve the output shapes.

    Symbols bind into a single table shared across *all* arguments, so an output
    that depends on a dim introduced by a later argument (``a:(M,K) @ b:(K,N)``)
    resolves correctly.

    Sizes are propagated exactly as received and never coerced with ``int()``, so
    ``SymInt`` dims from ``torch.compile(dynamic=True)`` survive unspecialized.
    """
    if len(input_shapes) != len(input_decls):
        raise ShapeResolutionError(
            f"{kernel_name}: expected {len(input_decls)} argument(s) to match the "
            f"spec, got {len(input_shapes)}"
        )

    symbols: dict[str, Any] = {}
    for (arg_name, declared), actual in zip(input_decls, input_shapes, strict=True):
        if len(actual) != len(declared):
            raise ShapeResolutionError(
                f"{kernel_name}: argument {arg_name!r} has rank {len(actual)}, but "
                f"the spec declares rank {len(declared)} {declared}"
            )
        for dim, size in zip(declared, actual, strict=True):
            literal = _literal_dim(dim)
            if literal is not None:
                if size != literal:
                    raise ShapeResolutionError(
                        f"{kernel_name}: argument {arg_name!r} declares literal dim "
                        f"{literal}, but got {size}"
                    )
                continue
            if dim not in symbols:
                symbols[dim] = size
            elif symbols[dim] != size:
                raise ShapeResolutionError(
                    f"{kernel_name}: symbol {dim!r} is bound to {symbols[dim]} but "
                    f"argument {arg_name!r} requires {size}"
                )

    resolved: list[tuple[Any, ...]] = []
    for out_name, declared in output_decls:
        dims: list[Any] = []
        for dim in declared:
            literal = _literal_dim(dim)
            if literal is not None:
                dims.append(literal)
                continue
            if dim not in symbols:
                raise ShapeResolutionError(
                    f"{kernel_name}: output {out_name!r} uses symbol {dim!r}, which "
                    f"no input binds; its shape cannot be derived"
                )
            dims.append(symbols[dim])
        resolved.append(tuple(dims))
    return resolved


def _decls(args: Sequence[Any]) -> list[ShapeDecl]:
    return [(arg.name, tuple(arg.shape)) for arg in args]


def resolve_output_shapes(
    spec: KernelSpec,
    input_shapes: Sequence[Sequence[Any]],
) -> list[tuple[Any, ...]]:
    """Resolve concrete output shapes for ``spec`` from concrete input shapes."""
    return _resolve_shapes(spec.name, _decls(spec.inputs), _decls(spec.outputs), input_shapes)


def make_fake_impl(spec: KernelSpec) -> Callable[..., Any]:
    """Build the fake (meta) implementation of ``cuda_engine::forward`` for ``spec``.

    Register with::

        torch.library.register_fake("cuda_engine::forward", make_fake_impl(spec))
    """
    import torch

    output_dtypes = [getattr(torch, TORCH_DTYPE_NAMES[out.dtype]) for out in spec.outputs]

    def fake_forward(*args: Any) -> Any:
        if not args:
            raise ShapeResolutionError(
                f"{spec.name}: fake implementation needs at least one tensor argument "
                f"to infer the output device"
            )
        shapes = resolve_output_shapes(spec, [tuple(arg.shape) for arg in args])
        outputs = [
            args[0].new_empty(shape, dtype=dtype)
            for shape, dtype in zip(shapes, output_dtypes, strict=True)
        ]
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    return fake_forward


_RENDER_TEMPLATE = '''\
"""Fake (meta) implementation for ``{qualname}`` -- kernel {kernel_name!r}.

Auto-generated by cuda-engine. Do not edit.

Registering this makes the kernel traceable by ``torch.compile`` (no graph break)
and usable with ``torch.export`` / AOTInductor. Without it the op is opaque to
Dynamo and splits the compiled region.

    import torch
    torch.ops.load_library("kernel.so")
    from .fake_impl import register
    register()
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

#: Shape declarations as plain data: ``(arg_name, symbolic_dims)``.
ShapeDecl = tuple[str, tuple[str, ...]]

KERNEL_NAME = {kernel_name!r}
QUALNAME = {qualname!r}
INPUT_DECLS = {input_decls!r}
OUTPUT_DECLS = {output_decls!r}
OUTPUT_DTYPES = {output_dtypes!r}


{exception_src}

{literal_dim_src}

{resolve_src}

def resolve_output_shapes(input_shapes: Any) -> list[tuple[Any, ...]]:
    """Resolve concrete output shapes from concrete input shapes."""
    return _resolve_shapes(KERNEL_NAME, INPUT_DECLS, OUTPUT_DECLS, input_shapes)


def fake_forward(*args: Any) -> Any:
    if not args:
        raise ShapeResolutionError(
            f"{{KERNEL_NAME}}: fake implementation needs at least one tensor "
            f"argument to infer the output device"
        )
    shapes = resolve_output_shapes([tuple(arg.shape) for arg in args])
    outputs = [
        args[0].new_empty(shape, dtype=getattr(torch, name))
        for shape, name in zip(shapes, OUTPUT_DTYPES, strict=True)
    ]
    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)


def register(qualname: str = QUALNAME) -> None:
    """Register ``fake_forward`` as the meta implementation of the op."""
    torch.library.register_fake(qualname, fake_forward)
'''


def render_fake_module(spec: KernelSpec, *, qualname: str = "cuda_engine::forward") -> str:
    """Render the fake implementation as standalone Python source.

    The result imports only ``torch`` -- never this package -- so it can be shipped
    inside an exported kernel package. The shape logic is embedded from this
    module's own source, so there is exactly one implementation of it.
    """
    return _RENDER_TEMPLATE.format(
        kernel_name=spec.name,
        qualname=qualname,
        input_decls=_decls(spec.inputs),
        output_decls=_decls(spec.outputs),
        output_dtypes=[TORCH_DTYPE_NAMES[out.dtype] for out in spec.outputs],
        exception_src=inspect.getsource(ShapeResolutionError).strip(),
        literal_dim_src=inspect.getsource(_literal_dim).strip(),
        resolve_src=inspect.getsource(_resolve_shapes).strip(),
    )
