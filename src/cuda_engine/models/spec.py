from __future__ import annotations

from enum import StrEnum
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

DType: TypeAlias = Literal["fp32", "fp16", "bf16", "fp64", "int32", "int64", "uint8", "int8"]
TargetArch: TypeAlias = Literal["sm_80", "sm_90", "sm_100", "sm_120"]


class OptimizationPriority(StrEnum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    BALANCED = "balanced"


class TensorArg(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    dtype: DType
    shape: tuple[str, ...] = Field(description="Symbolic shape, e.g. ('B', 'S', 'D')")
    layout_hint: Literal["row_major", "col_major", "any"] = "any"


class PrecisionTolerance(BaseModel):
    model_config = ConfigDict(frozen=True)

    rtol: float = 1e-3
    atol: float = 1e-3


class KernelSpec(BaseModel):
    """Frozen Stage 1 contract; downstream stages must not mutate it."""

    model_config = ConfigDict(frozen=True)

    name: str
    target_arch: TargetArch
    inputs: list[TensorArg]
    outputs: list[TensorArg]
    precision_tolerance: PrecisionTolerance
    optimization_priority: OptimizationPriority
    notes: str = ""
