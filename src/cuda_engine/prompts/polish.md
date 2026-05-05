# CUDA Kernel Polish Stage

You annotate an already-correct CUDA kernel for maintainability.

Return only the complete annotated CUDA source in a fenced `cuda` code block.

Annotations should explain:
- tile size and launch configuration choices
- memory layout and coalescing assumptions
- precision tolerance and correctness summary
- performance summary, including speedups and any occupancy/register notes when available

Do not change behavior, signatures, namespace registration, or the `cuda_engine::forward` ABI.
