"""KernelBench filter + adapter.

External eval suite based on
https://github.com/ScalingIntelligence/KernelBench.

The filter scans a local KernelBench checkout, classifies each Model
file's `forward` method by AST inspection, and reports which ones fall
within cuda-engine v1's scope (elementwise + reductions). Out-of-scope
ops (matmul, attention, conv) are excluded.

See `evals/kernelbench/README.md` for the fetch + filter workflow.
"""

from evals.kernelbench.filter import (
    KernelClassification,
    classify_kernelbench_file,
    classify_kernelbench_tree,
)

__all__ = [
    "KernelClassification",
    "classify_kernelbench_file",
    "classify_kernelbench_tree",
]
