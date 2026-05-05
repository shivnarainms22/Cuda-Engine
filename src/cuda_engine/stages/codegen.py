import re
from typing import Any

from cuda_engine.models import KernelArtifact, KernelSpec
from cuda_engine.prompts import load_prompt
from cuda_engine.services.gpu.base import CompileResult
from cuda_engine.services.llm.tools import COMPILE_KERNEL
from cuda_engine.stages.base import BudgetExhaustedError, Stage
from cuda_engine.targets import load_target_caps


class Stage2Codegen(Stage):
    name = "codegen"

    def run(self, *, spec: KernelSpec, run_id: str, retry_budget: int = 3) -> KernelArtifact:
        if self.llm is None or self.gpu is None or self.store is None:
            raise RuntimeError("Stage2Codegen requires llm, gpu, and store services")

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    "Generate kernel.cu for this KernelSpec, then call compile_kernel.\n\n"
                    f"{spec.model_dump_json(indent=2)}"
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

        last_result: CompileResult | None = None
        for attempt in range(1, retry_budget + 1):
            response = self.llm.complete(
                system=system,
                messages=messages,
                tools=[COMPILE_KERNEL],
                model="claude-sonnet-4-6",
            )
            src = _source_from_response(response.text, response.tool_calls)
            attempt_dir = f"stage2_codegen/attempt_{attempt:02d}"
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
                final_kernel = self.store.write_text(run_id, "stage2_codegen/final/kernel.cu", compile_src)
                final_so = last_result.so_path
                if last_result.so_path is not None:
                    if last_result.so_path.exists():
                        final_so = self.store.write_bytes(
                            run_id,
                            "stage2_codegen/final/kernel.so",
                            last_result.so_path.read_bytes(),
                        )
                    else:
                        self.store.write_text(
                            run_id,
                            "stage2_codegen/final/kernel.so.path",
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

        raise BudgetExhaustedError(
            f"codegen exhausted retry budget after {retry_budget} attempts: "
            f"{_exhausted_budget_detail(last_result)}"
        )


def _compile_call(tool_calls: list[dict[str, Any]]) -> dict[str, Any] | None:
    for call in tool_calls:
        if call.get("name") == "compile_kernel":
            return call
    return None


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
