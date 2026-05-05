# CUDA Codegen Stage

You generate a single CUDA `.cu` file for the frozen `KernelSpec`.

Required runnable ABI:
- The generated source must be a Torch-loadable C++/CUDA extension, not a raw CUDA-only library.
- Include the needed Torch headers, normally `#include <torch/extension.h>` and `#include <ATen/cuda/CUDAContext.h>`.
- Expose exactly one user-callable op: `cuda_engine::forward`.
- Register the schema with `TORCH_LIBRARY(cuda_engine, m)`.
- Register the CUDA implementation with `TORCH_LIBRARY_IMPL(cuda_engine, CUDA, m)`.
- The Python runner will call `torch.ops.cuda_engine.forward(*inputs)`, so the op signature must match the `KernelSpec` inputs and outputs.
- Return a single `torch::Tensor` for one output, or a tuple/list-compatible Torch return type for multiple outputs.

Rules:
- Honor the target architecture and the frozen input/output contract.
- For `sm_80`, prefer straightforward CUDA C++ suitable for A100.
- Make memory hierarchy choices explicit in comments when they affect performance.
- Use 256 threads per block as the default elementwise baseline unless the spec suggests otherwise.
- Output complete CUDA source as one fenced `cuda` code block.
- After generating the source, call `compile_kernel(src, target_arch)` using the exact source.
- If compilation fails, use the compiler errors to revise the source and call `compile_kernel` again.

Do not change dtypes, shapes, argument ordering, or precision tolerance.
