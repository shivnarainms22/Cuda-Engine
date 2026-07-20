import re
from typing import Any

from cuda_engine.models import CorrectnessReport, KernelArtifact, KernelSpec
from cuda_engine.prompts import load_prompt
from cuda_engine.services.gpu.base import CompileResult
from cuda_engine.services.llm.tools import COMPILE_KERNEL
from cuda_engine.stages.base import BudgetExhaustedError, SonnetFailureSummary, Stage
from cuda_engine.targets import load_target_caps


class Stage2Codegen(Stage):
    name = "codegen"

    def run(
        self,
        *,
        spec: KernelSpec,
        run_id: str,
        model: str,
        retry_budget: int = 3,
        repair_context: CorrectnessReport | None = None,
        escalation_context: SonnetFailureSummary | None = None,
        artifact_prefix: str = "stage2_codegen",
    ) -> KernelArtifact:
        if self.llm is None or self.gpu is None or self.store is None:
            raise RuntimeError("Stage2Codegen requires llm, gpu, and store services")

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": _initial_user_prompt(
                    spec=spec,
                    repair_context=repair_context,
                    escalation_context=escalation_context,
                ),
            }
        ]
        system = [
            {
                "type": "text",
                "text": load_prompt("codegen"),
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": f"Target capabilities:\n{load_target_caps(spec.target_arch)}",
                "cache_control": {"type": "ephemeral"},
            },
        ]
        # For fp16/bf16 matrix multiplies on a tensor-core target, add a correct
        # WMMA reference — the LLM knows tensor cores exist but often can't compile
        # the WMMA API from memory (the Rung 1 bottleneck). Only for matmul-shaped
        # specs so elementwise/reduction prompts stay focused.
        tc_guidance = _tensor_core_matmul_guidance(spec)
        if tc_guidance is not None:
            system.append(
                {"type": "text", "text": tc_guidance, "cache_control": {"type": "ephemeral"}}
            )

        last_result: CompileResult | None = None
        last_src: str = ""
        for attempt in range(1, retry_budget + 1):
            response = self.llm.complete(
                system=system,
                messages=messages,
                tools=[COMPILE_KERNEL],
                model=model,
            )
            src = _source_from_response(response.text, response.tool_calls)
            attempt_dir = f"{artifact_prefix}/attempt_{attempt:02d}"
            kernel_path = self.store.write_text(run_id, f"{attempt_dir}/kernel.cu", src)
            self.store.write_text(run_id, f"{attempt_dir}/llm_response.md", response.text)

            compile_call = _compile_call(response.tool_calls)
            if compile_call is None:
                messages.append(
                    {
                        "role": "user",
                        "content": "You must call compile_kernel with the generated CUDA source.",
                    }
                )
                continue

            compile_input = compile_call.get("input", {})
            compile_src = str(compile_input.get("src") or src)
            last_src = compile_src
            target_arch = str(compile_input.get("target_arch") or spec.target_arch)
            extra_flags = tuple(str(flag) for flag in compile_input.get("extra_flags", ()))
            last_result = self.gpu.compile(
                compile_src,
                target_arch=target_arch,
                extra_flags=extra_flags,
            )
            self.store.write_text(run_id, f"{attempt_dir}/compile.log", last_result.log)
            self.store.write_text(run_id, f"{attempt_dir}/compile_log.txt", last_result.log)
            self.store.write_json(run_id, f"{attempt_dir}/result.json", last_result)

            if last_result.ok:
                final_kernel = self.store.write_text(run_id, f"{artifact_prefix}/final/kernel.cu", compile_src)
                final_so = last_result.so_path
                if last_result.so_path is not None:
                    if last_result.so_path.exists():
                        final_so = self.store.write_bytes(
                            run_id,
                            f"{artifact_prefix}/final/kernel.so",
                            last_result.so_path.read_bytes(),
                        )
                    else:
                        self.store.write_text(
                            run_id,
                            f"{artifact_prefix}/final/kernel.so.path",
                            str(last_result.so_path),
                        )
                return KernelArtifact(
                    kernel_cu_path=final_kernel or kernel_path,
                    kernel_so_path=final_so,
                    compile_log=last_result.log,
                    ptx_size_bytes=last_result.ptx_size_bytes,
                )

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Compilation failed. Fix kernel.cu and call compile_kernel again.\n\n"
                        f"Errors:\n{last_result.errors}\n\nCompile log:\n{last_result.log}"
                    ),
                }
            )

        errors_str = "" if last_result is None else "\n".join(last_result.errors)
        log_str = "" if last_result is None else last_result.log
        summary = SonnetFailureSummary(
            last_compile_errors=errors_str,
            last_compile_log=log_str,
            last_source_attempt=last_src,
            attempts_made=retry_budget,
        )
        raise BudgetExhaustedError(
            f"codegen exhausted retry budget after {retry_budget} attempts: "
            f"{_exhausted_budget_detail(last_result)}",
            summary=summary,
        )


def _is_matmul_spec(spec: KernelSpec) -> bool:
    """Heuristic: a matmul has at least two rank-2 (matrix) inputs. Elementwise
    and reduction kernels do not, so they never trigger the WMMA guidance."""
    return sum(1 for arg in spec.inputs if len(arg.shape) == 2) >= 2


def _tensor_core_matmul_guidance(spec: KernelSpec) -> str | None:
    """Return the WMMA reference iff this is a matmul whose input dtype has
    tensor-core support on the target (fp16/bf16 on sm_80+); else None."""
    if not _is_matmul_spec(spec):
        return None
    tensor_cores = load_target_caps(spec.target_arch).get("tensor_cores", {})
    if any(arg.dtype in tensor_cores for arg in spec.inputs):
        return _WMMA_GUIDANCE
    return None


# Correctness-focused WMMA reference (compiling is the bottleneck, not tuning).
# Framed permissively: the kernel MAY use CUDA cores instead, but if it uses
# tensor cores it must use this API correctly.
_WMMA_GUIDANCE = """\
Tensor-core (WMMA) reference for fp16/bf16 matrix multiply on this target.
Using tensor cores is recommended for throughput, but getting the WMMA API
slightly wrong is the usual cause of compile failures — follow this exactly:

- Includes: `#include <cuda_fp16.h>` and `#include <mma.h>`; then `using namespace nvcuda;`.
- Fragments (16x16x16 tile for fp16/bf16), with EXACT template arguments:
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;   // accumulate in fp32
- LAYOUT: inputs are ROW-MAJOR and contiguous, so BOTH fragments are row_major,
  with lda = K and ldb = N. Do not copy the col_major matrix_b from generic WMMA
  samples — those assume a column-major B. A layout mismatch still COMPILES and
  silently computes A @ B.T, so it will not show up as a compile error.
- Reinterpret fp16 tensor pointers as `const half*` / `half*`.
- One warp computes one 16x16 output tile. Per tile, loop K in steps of 16:
    wmma::fill_fragment(acc_frag, 0.0f);
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a_frag, aPtr + (tileRow * 16) * lda + k, lda);
        wmma::load_matrix_sync(b_frag, bPtr + k * ldb + tileCol * 16, ldb);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
- OUTPUT DTYPE: store_matrix_sync requires the pointer element type to match the
  fragment type, so a float accumulator CANNOT be stored to a half* output — that
  is a compile error, not a warning. For an fp16 output, stage through shared
  memory and convert. The staging buffer MUST be per-warp — every warp in the
  block reaches this code, so a single shared tile is a data race:
    const int warpId = threadIdx.x / 32;      // WARPS_PER_BLOCK = blockDim.x / 32
    const int laneId = threadIdx.x % 32;
    __shared__ float stage[WARPS_PER_BLOCK][16 * 16];
    wmma::store_matrix_sync(&stage[warpId][0], acc_frag, 16, wmma::mem_row_major);
    __syncwarp();
    for (int i = laneId; i < 16 * 16; i += 32) {
        int r = i / 16, c = i % 16;
        int gr = tileRow * 16 + r, gc = tileCol * 16 + c;
        if (gr < M && gc < N) cPtr[gr * ldc + gc] = __float2half(stage[warpId][i]);
    }
  (For an fp32 output you may store the accumulator straight to cPtr with ldc.)
- Launch with blockDim.x a multiple of 32 (whole warps).
- SHAPES: a remainder path is REQUIRED, not optional. Correctness is checked at
  N = 0, 1, 127, 128, 1024, 4097 and at the 4096 benchmark shape — so 1, 127 and
  4097 are not multiples of 16. A WMMA-only kernel fails the correctness gate.
  Handle ragged edges by guarding the stores as above and zero-padding the loaded
  tiles through shared memory, or by routing sizes below one 16x16 tile to a
  scalar path. N = 0 must be a safe no-op.
- If unsure whether WMMA will compile, a correct shared-memory-tiled CUDA-core
  kernel is an acceptable fallback — correctness first."""


def _compile_call(tool_calls: list[dict[str, Any]]) -> dict[str, Any] | None:
    for call in tool_calls:
        if call.get("name") == "compile_kernel":
            return call
    return None


def _initial_user_prompt(
    *,
    spec: KernelSpec,
    repair_context: CorrectnessReport | None,
    escalation_context: SonnetFailureSummary | None = None,
) -> str:
    base = (
        "Generate kernel.cu for this KernelSpec, then call compile_kernel.\n\n"
        f"{spec.model_dump_json(indent=2)}"
        if repair_context is None
        else (
            "Repair kernel.cu for this KernelSpec. The previous kernel compiled but failed "
            "correctness. Use the correctness report to fix the implementation, then call "
            "compile_kernel with the repaired CUDA source.\n\n"
            f"KernelSpec:\n{spec.model_dump_json(indent=2)}\n\n"
            f"Correctness report:\n{repair_context.model_dump_json(indent=2)}"
        )
    )
    if escalation_context is None:
        return base
    return f"{_escalation_preamble(escalation_context)}\n\n{base}"


def _escalation_preamble(summary: SonnetFailureSummary) -> str:
    return (
        f"Previous {summary.attempts_made} compile attempts failed. "
        "Address the underlying issue rather than repeating the prior approach.\n\n"
        f"Last compile errors:\n{summary.last_compile_errors}\n\n"
        f"Last compile log:\n{summary.last_compile_log}\n\n"
        f"Last source attempt:\n```cuda\n{summary.last_source_attempt}\n```"
    )


def _source_from_response(text: str, tool_calls: list[dict[str, Any]]) -> str:
    call = _compile_call(tool_calls)
    if call is not None:
        src = call.get("input", {}).get("src")
        if src:
            return str(src)
    match = re.search(r"```(?:cuda|cpp|c\+\+)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _exhausted_budget_detail(last_result: CompileResult | None) -> str:
    if last_result is None:
        return "no compile result"
    return f"errors={last_result.errors}; compile_log={last_result.log}"
