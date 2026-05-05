from collections.abc import Callable
from typing import Any, cast

from cuda_engine.models import KernelSpec, OptimizationPriority, PrecisionTolerance, TensorArg
from cuda_engine.models.spec import TargetArch
from cuda_engine.stages.base import Stage


class Stage1Interview(Stage):
    name = "interview"

    def run(
        self,
        *,
        prompt: str,
        reference: Callable[..., Any],
        target_arch: str,
        run_id: str,
    ) -> KernelSpec:
        return KernelSpec(
            name="placeholder",
            target_arch=cast(TargetArch, target_arch),
            inputs=[TensorArg(name="x", dtype="fp32", shape=("N",))],
            outputs=[TensorArg(name="out", dtype="fp32", shape=("N",))],
            precision_tolerance=PrecisionTolerance(rtol=1e-3, atol=1e-3),
            optimization_priority=OptimizationPriority.BALANCED,
        )
