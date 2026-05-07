from collections.abc import Callable
from typing import Any

from cuda_engine.models import CorrectnessReport, KernelArtifact, KernelSpec, PerformanceReport
from cuda_engine.prompts import load_prompt
from cuda_engine.stages.base import Stage
from cuda_engine.stages.correctness import Stage3Correctness


class Stage5Polish(Stage):
    name = "polish"

    def run(
        self,
        *,
        spec: KernelSpec,
        artifact: KernelArtifact,
        correctness: CorrectnessReport,
        performance: PerformanceReport,
        reference: Callable[..., Any],
        run_id: str,
    ) -> KernelArtifact:
        if self.llm is None or self.gpu is None or self.store is None:
            raise RuntimeError("Stage5Polish requires llm, gpu, and store services")

        source = _read_artifact_source(artifact, run_id, self.store)
        user_content = (
            "Annotate this verified CUDA kernel without changing behavior.\n\n"
            f"KernelSpec:\n{spec.model_dump_json(indent=2)}\n\n"
            f"CorrectnessReport:\n{correctness.model_dump_json(indent=2)}\n\n"
            f"PerformanceReport:\n{performance.model_dump_json(indent=2)}\n\n"
            f"Source:\n```cuda\n{source}\n```"
        )
        response = self.llm.complete(
            system=[
                {
                    "type": "text",
                    "text": load_prompt("polish"),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
            tools=None,
            model="claude-sonnet-4-6",
        )
        annotated = _extract_cuda_source(response.text)
        annotated_path = self.store.write_text(
            run_id,
            "stage5_polish/kernel_annotated.cu",
            annotated,
        )
        self.store.write_text(run_id, "stage5_polish/llm_response.md", response.text)

        compile_result = self.gpu.compile(annotated, target_arch=spec.target_arch)
        self.store.write_text(run_id, "stage5_polish/compile.log", compile_result.log)
        self.store.write_json(run_id, "stage5_polish/compile_result.json", compile_result)
        if not compile_result.ok or compile_result.so_path is None:
            self.store.write_json(
                run_id,
                "stage5_polish/status.json",
                {
                    "accepted": False,
                    "reason": "compile failed",
                    "kernel_cu_path": str(annotated_path),
                },
            )
            return artifact

        candidate = KernelArtifact(
            kernel_cu_path=annotated_path,
            kernel_so_path=compile_result.so_path,
            compile_log=compile_result.log,
            ptx_size_bytes=compile_result.ptx_size_bytes,
        )
        polish_correctness = Stage3Correctness(gpu=self.gpu, store=self.store).run(
            spec=spec,
            artifact=candidate,
            reference=reference,
            run_id=run_id,
            artifact_prefix="stage5_polish/correctness",
        )
        self.store.write_json(
            run_id,
            "stage5_polish/correctness_report.json",
            polish_correctness.model_dump(mode="json"),
        )
        if not polish_correctness.passed:
            self.store.write_json(
                run_id,
                "stage5_polish/status.json",
                {
                    "accepted": False,
                    "reason": "correctness failed",
                    "kernel_cu_path": str(annotated_path),
                },
            )
            return artifact

        final_kernel = self.store.write_text(run_id, "stage5_polish/final/kernel.cu", annotated)
        final_so = compile_result.so_path
        if compile_result.so_path.exists():
            final_so = self.store.write_bytes(
                run_id,
                "stage5_polish/final/kernel.so",
                compile_result.so_path.read_bytes(),
            )
        else:
            self.store.write_text(
                run_id,
                "stage5_polish/final/kernel.so.path",
                str(compile_result.so_path),
            )
        self.store.write_json(
            run_id,
            "stage5_polish/status.json",
            {
                "accepted": True,
                "reason": "validated",
                "kernel_cu_path": str(final_kernel),
                "kernel_so_path": str(final_so),
            },
        )
        return KernelArtifact(
            kernel_cu_path=final_kernel,
            kernel_so_path=final_so,
            compile_log=compile_result.log,
            ptx_size_bytes=compile_result.ptx_size_bytes,
        )


def _extract_cuda_source(text: str) -> str:
    import re

    match = re.search(r"```(?:cuda|cpp|c\+\+)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _read_artifact_source(artifact: KernelArtifact, run_id: str, store: object) -> str:
    path_key = artifact.kernel_cu_path.as_posix().replace("\\", "/")
    marker = f"<memory>/{run_id}/"
    if marker in path_key:
        rel_path = path_key.split(marker, 1)[1]
        files = getattr(store, "_files", None)
        if files is not None:
            content = files[(run_id, rel_path)]
            return str(content.decode())
    return artifact.kernel_cu_path.read_text(encoding="utf-8")
