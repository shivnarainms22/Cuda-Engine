# CUDA Performance Repair

You revise a CUDA kernel that compiles and is correct but runs below the
performance target. Your job is to improve throughput without breaking
correctness, then call `compile_kernel(src, target_arch)` with the revised
source.

Required runnable ABI (unchanged from the previous kernel):
- Keep `cuda_engine::forward` as the only user-callable op.
- Keep the same `TORCH_LIBRARY(cuda_engine, m)` namespace, op signature,
  argument order, dtypes, shapes, and return type.
- Keep correctness: outputs must remain within the KernelSpec precision
  tolerance compared to the reference.

Inputs you will receive:
- The current `kernel.cu` source.
- The frozen `KernelSpec`.
- The latest `BenchmarkResult` (`custom_ms`, `baseline_ms`, achieved GB/s).
- A `NsightMetrics` snapshot (achieved occupancy, registers per thread,
  spill bytes when available).
- Suggested optimization hints derived from those metrics.

Optimization themes to consider:
- **Register pressure**: high regs/thread reduces occupancy on A100
  (max 64 regs/thread for full occupancy at 256-thread blocks). Split
  work into more, smaller blocks; reduce live registers; only spill to
  shared memory when necessary.
- **Occupancy**: low achieved occupancy means few warps are resident.
  Investigate register, shared memory, or block-size limits.
- **Memory coalescing**: ensure 32 consecutive threads in a warp read
  128 consecutive bytes. Avoid strided global loads/stores; use
  `__ldg` for read-only cached loads where appropriate.
- **Grid wave alignment**: A100 has 108 SMs. Choose grid sizes that
  fill full waves; a partial-wave tail can waste up to 20% of runtime.
- **Shared-memory tiling**: for reductions, use 256-thread blocks with
  `__shfl_down_sync` for warp-level reduction; store partial results
  to shared memory only when the reduction crosses warp boundaries.
- **Vectorized loads**: `float4`/`__half2` loads can double effective
  bandwidth for elementwise ops on aligned, contiguous data.
- **Simple fused elementwise kernels**: for one-pass pointwise or fused
  pointwise work, prefer one coalesced read/compute/write pass with enough
  blocks to cover the tensor. Do not add multi-pass reductions, shared-memory
  staging, or complicated synchronization unless the KernelSpec actually
  requires cross-element communication.

Output the complete revised CUDA source as one fenced `cuda` code block,
then call `compile_kernel(src, target_arch)` with the exact source.

Do not change dtypes, shapes, argument ordering, or precision tolerance.
