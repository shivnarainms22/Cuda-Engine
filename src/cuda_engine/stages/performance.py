from collections.abc import Callable
from typing import Any

from cuda_engine.config import SynthesisConfig
from cuda_engine.models import KernelArtifact, KernelSpec, PerformanceReport
from cuda_engine.prompts import load_prompt
from cuda_engine.services.gpu.base import BenchmarkResult, GPURunner, NsightMetrics
from cuda_engine.services.llm.base import LLMClient
from cuda_engine.services.llm.tools import COMPILE_KERNEL
from cuda_engine.services.store.base import ArtifactStore
from cuda_engine.stages.base import Stage
from cuda_engine.stages.codegen import _source_from_response
from cuda_engine.stages.correctness import _make_inputs
from cuda_engine.stages.polish import _read_artifact_source


class Stage4Performance(Stage):
    name = "performance"

    def __init__(
        self,
        llm: LLMClient | None = None,
        gpu: GPURunner | None = None,
        store: ArtifactStore | None = None,
        cfg: SynthesisConfig | None = None,
    ) -> None:
        super().__init__(llm=llm, gpu=gpu, store=store)
        self.cfg = cfg or SynthesisConfig()

    def run(
        self,
        *,
        spec: KernelSpec,
        artifact: KernelArtifact,
        run_id: str,
        retry_budget: int = 3,
        reference: Callable[..., Any] | None = None,
    ) -> tuple[PerformanceReport, KernelArtifact]:
        if self.gpu is None or self.store is None:
            raise RuntimeError("Stage4Performance requires gpu and store services")
        if artifact.kernel_so_path is None:
            report = PerformanceReport(
                speedup_vs_reference=None,
                speedup_vs_torch_compile=None,
                below_target=True,
                notes=["kernel_so_path is required for performance benchmarking"],
            )
            _write_report(self.store, run_id, report)
            return report, artifact

        benchmark_shape = _benchmark_shape(spec, total_elements=self.cfg.performance_shape_n)
        inputs = _make_inputs(spec, shape=benchmark_shape)
        benchmark = self.gpu.benchmark_kernel(
            artifact.kernel_so_path,
            inputs,
            reference=reference,
            warmup_iterations=self.cfg.benchmark_warmup_iterations,
            timed_iterations=self.cfg.benchmark_timed_iterations,
        )
        self.store.write_json(
            run_id,
            "stage4_performance/benchmark.json",
            _benchmark_payload(benchmark, cfg=self.cfg),
        )

        if not benchmark.ok:
            report = PerformanceReport(
                speedup_vs_reference=None,
                speedup_vs_torch_compile=None,
                below_target=True,
                notes=[benchmark.stderr or "benchmark failed"],
            )
            _write_report(self.store, run_id, report)
            return report, artifact

        cached_baseline_ms = benchmark.baseline_ms
        speedup = _speedup(baseline_ms=cached_baseline_ms, custom_ms=benchmark.custom_ms)
        target = self.cfg.perf_target_speedup_vs_torch_compile
        warnings: list[str] = []
        notes: list[str] = []
        if benchmark.baseline_error is not None:
            warnings.append(benchmark.baseline_error)

        current_artifact = artifact
        current_benchmark = benchmark
        current_speedup = speedup

        if self.llm is not None and retry_budget > 0:
            current_artifact, current_benchmark, current_speedup, retry_warnings, retry_notes = self._retry_loop(
                spec=spec,
                artifact=current_artifact,
                benchmark=current_benchmark,
                speedup=current_speedup,
                target=target,
                inputs=inputs,
                run_id=run_id,
                retry_budget=retry_budget,
                model=self.cfg.sonnet_model,
                attempt_offset=0,
                baseline_ms=cached_baseline_ms,
            )
            warnings.extend(retry_warnings)
            notes.extend(retry_notes)

        if (
            (current_speedup is None or current_speedup < target)
            and self.cfg.escalate_to_opus_on_bust
            and self.cfg.opus_retry_budget_performance > 0
            and self.llm is not None
        ):
            notes.append(
                f"escalated to opus after sonnet retry budget exhausted at speedup "
                f"{_fmt_speedup(current_speedup)}"
            )
            (
                current_artifact,
                current_benchmark,
                current_speedup,
                opus_warnings,
                opus_notes,
            ) = self._retry_loop(
                spec=spec,
                artifact=current_artifact,
                benchmark=current_benchmark,
                speedup=current_speedup,
                target=target,
                inputs=inputs,
                run_id=run_id,
                retry_budget=self.cfg.opus_retry_budget_performance,
                model=self.cfg.opus_model,
                attempt_offset=retry_budget,
                baseline_ms=cached_baseline_ms,
            )
            warnings.extend(opus_warnings)
            notes.extend(opus_notes)

        report = PerformanceReport(
            speedup_vs_reference=current_speedup,
            speedup_vs_torch_compile=current_speedup,
            achieved_gbps=current_benchmark.achieved_gbps,
            below_target=current_speedup is None or current_speedup < target,
            warnings=warnings,
            notes=notes,
        )
        _write_report(self.store, run_id, report)
        return report, current_artifact

    def _retry_loop(
        self,
        *,
        spec: KernelSpec,
        artifact: KernelArtifact,
        benchmark: BenchmarkResult,
        speedup: float | None,
        target: float,
        inputs: list[Any],
        run_id: str,
        retry_budget: int,
        model: str,
        attempt_offset: int = 0,
        baseline_ms: float | None,
    ) -> tuple[KernelArtifact, BenchmarkResult, float | None, list[str], list[str]]:
        assert self.llm is not None
        assert self.gpu is not None
        assert self.store is not None

        warnings: list[str] = []
        notes: list[str] = []
        current_artifact = artifact
        current_benchmark = benchmark
        current_speedup = speedup
        best_artifact = current_artifact
        best_benchmark = current_benchmark
        best_speedup = current_speedup
        system = [
            {
                "type": "text",
                "text": load_prompt("perf_fix"),
                "cache_control": {"type": "ephemeral"},
            }
        ]

        for local_attempt in range(1, retry_budget + 1):
            attempt = local_attempt + attempt_offset
            if current_artifact.kernel_so_path is None:
                warnings.append(f"perf_repair attempt {attempt}: missing kernel_so_path")
                break

            metrics = self.gpu.profile(current_artifact.kernel_so_path, inputs)
            hints = _format_perf_hints(metrics, benchmark=current_benchmark)
            try:
                src = _read_artifact_source(current_artifact, run_id, self.store)
            except (FileNotFoundError, OSError) as exc:
                warnings.append(f"perf_repair attempt {attempt}: source unreadable ({exc})")
                break

            attempt_dir = f"stage4_performance/perf_repair/attempt_{attempt:02d}"
            self.store.write_json(
                run_id,
                f"{attempt_dir}/nsight.json",
                metrics.model_dump(mode="json"),
            )
            self.store.write_json(
                run_id,
                f"{attempt_dir}/benchmark.json",
                _benchmark_payload(current_benchmark, cfg=self.cfg),
            )

            user_message = _build_perf_repair_user_message(
                spec=spec,
                src=src,
                benchmark=current_benchmark,
                metrics=metrics,
                hints=hints,
                speedup=current_speedup,
                target=target,
            )
            self.store.write_text(
                run_id,
                f"{attempt_dir}/prompt_to_llm.md",
                user_message,
            )

            response = self.llm.complete(
                system=system,
                messages=[{"role": "user", "content": user_message}],
                tools=[COMPILE_KERNEL],
                model=model,
            )
            self.store.write_text(run_id, f"{attempt_dir}/llm_response.md", response.text)

            new_src = _extract_source_from_response(response)
            if new_src is None:
                warnings.append(f"perf_repair attempt {attempt}: LLM did not return CUDA source")
                continue

            self.store.write_text(run_id, f"{attempt_dir}/kernel.cu", new_src)
            compile_result = self.gpu.compile(new_src, target_arch=spec.target_arch)
            self.store.write_text(run_id, f"{attempt_dir}/compile.log", compile_result.log)
            if not compile_result.ok or compile_result.so_path is None:
                warnings.append(f"perf_repair attempt {attempt}: compile failed")
                continue

            persisted_so = self.store.write_bytes(
                run_id,
                f"{attempt_dir}/kernel.so",
                compile_result.so_path.read_bytes()
                if compile_result.so_path.exists()
                else b"",
            )
            persisted_cu = self.store.write_text(
                run_id,
                f"{attempt_dir}/kernel.cu",
                new_src,
            )
            candidate_so = persisted_so or compile_result.so_path
            candidate = KernelArtifact(
                kernel_cu_path=persisted_cu,
                kernel_so_path=candidate_so,
                compile_log=compile_result.log,
                ptx_size_bytes=compile_result.ptx_size_bytes,
            )
            new_benchmark = self.gpu.benchmark_kernel(
                candidate_so,
                inputs,
                reference=None,
                warmup_iterations=self.cfg.benchmark_warmup_iterations,
                timed_iterations=self.cfg.benchmark_timed_iterations,
            )
            self.store.write_json(
                run_id,
                f"{attempt_dir}/benchmark_after.json",
                _benchmark_payload(new_benchmark, cfg=self.cfg),
            )
            if not new_benchmark.ok:
                warnings.append(f"perf_repair attempt {attempt}: benchmark failed after recompile")
                continue

            new_speedup = _speedup(
                baseline_ms=baseline_ms, custom_ms=new_benchmark.custom_ms
            )
            next_best = _max_optional(best_speedup, new_speedup)
            notes.append(
                f"perf_repair attempt {attempt}: speedup {_fmt_speedup(current_speedup)} -> "
                f"{_fmt_speedup(new_speedup)} (best={_fmt_speedup(next_best)})"
            )
            current_artifact = candidate
            current_benchmark = new_benchmark
            current_speedup = new_speedup
            if new_speedup is not None and (best_speedup is None or new_speedup > best_speedup):
                best_artifact = candidate
                best_benchmark = new_benchmark
                best_speedup = new_speedup

        if best_speedup is None or best_speedup < target:
            warnings.append(
                f"perf retry budget exhausted: best speedup {_fmt_speedup(best_speedup)} "
                f"below target {target:.3f}"
            )
        return best_artifact, best_benchmark, best_speedup, warnings, notes


def _speedup(*, baseline_ms: float | None, custom_ms: float) -> float | None:
    if baseline_ms is None or custom_ms <= 0:
        return None
    return baseline_ms / custom_ms


def _max_optional(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _fmt_speedup(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def _benchmark_shape(spec: KernelSpec, *, total_elements: int) -> tuple[int, ...]:
    rank = max((len(arg.shape) for arg in spec.inputs), default=1)
    if rank <= 1:
        return (total_elements,)
    dim = max(1, round(total_elements ** (1 / rank)))
    return tuple(dim for _ in range(rank))


def _write_report(store: ArtifactStore, run_id: str, report: PerformanceReport) -> None:
    store.write_json(run_id, "stage4_performance/report.json", report.model_dump(mode="json"))


def _benchmark_payload(benchmark: BenchmarkResult, *, cfg: SynthesisConfig) -> dict[str, Any]:
    payload = benchmark.model_dump(mode="json")
    payload["settings"] = {
        "performance_shape_n": cfg.performance_shape_n,
        "benchmark_warmup_iterations": cfg.benchmark_warmup_iterations,
        "benchmark_timed_iterations": cfg.benchmark_timed_iterations,
    }
    return payload


def _format_perf_hints(
    metrics: NsightMetrics, *, benchmark: BenchmarkResult
) -> list[str]:
    hints: list[str] = []
    if metrics.regs_per_thread is not None and metrics.regs_per_thread >= 64:
        hints.append(
            f"Register pressure is high ({metrics.regs_per_thread} regs/thread). "
            "Above 64 regs/thread caps occupancy on A100. Reduce live registers, "
            "split work across more blocks, or lower block size."
        )
    if metrics.occupancy is not None and metrics.occupancy < 0.5:
        hints.append(
            f"Achieved occupancy is {metrics.occupancy:.2f}. Investigate register pressure, "
            "shared memory usage, or block-size limits."
        )
    if metrics.spill_bytes > 0:
        hints.append(
            f"Spill bytes detected: {metrics.spill_bytes}. Local memory spills indicate "
            "register pressure beyond the file. Reduce live state."
        )
    if (
        metrics.uncoalesced_global_loads_pct is not None
        and metrics.uncoalesced_global_loads_pct > 20.0
    ):
        hints.append(
            f"Uncoalesced global loads at {metrics.uncoalesced_global_loads_pct:.1f}%. "
            "Restructure access patterns so 32 consecutive threads read 128 contiguous bytes."
        )
    achieved_gbps = metrics.achieved_bandwidth_gbps or benchmark.achieved_gbps
    if achieved_gbps is not None and achieved_gbps < 300.0:
        hints.append(
            f"Measured memory bandwidth is low ({achieved_gbps:.1f} GB/s). "
            "For pointwise or fused elementwise kernels, prefer a single coalesced pass with "
            "vectorized loads/stores such as float4 for fp32 or __half2 for fp16 when alignment "
            "and shape divisibility allow it."
        )
    if benchmark.baseline_ms is not None and benchmark.custom_ms > benchmark.baseline_ms:
        hints.append(
            f"Custom kernel ({benchmark.custom_ms:.3f}ms) is slower than the eager baseline "
            f"({benchmark.baseline_ms:.3f}ms). The current implementation is leaving the "
            "main bottleneck unresolved."
        )
    if not hints:
        hints.append(
            "No specific bottleneck flagged by basic Nsight metrics. Inspect the source "
            "for redundant work, suboptimal launch geometry, or missed vectorization."
        )
    return hints


def _build_perf_repair_user_message(
    *,
    spec: KernelSpec,
    src: str,
    benchmark: BenchmarkResult,
    metrics: NsightMetrics,
    hints: list[str],
    speedup: float | None,
    target: float,
) -> str:
    hints_block = "\n".join(f"- {hint}" for hint in hints)
    return (
        "Revise this CUDA kernel to improve performance, then call "
        "compile_kernel(src, target_arch).\n\n"
        f"Current speedup vs torch.compile: {_fmt_speedup(speedup)} (target: {target:.3f}).\n\n"
        f"KernelSpec:\n{spec.model_dump_json(indent=2)}\n\n"
        f"BenchmarkResult:\n{benchmark.model_dump_json(indent=2)}\n\n"
        f"NsightMetrics:\n{metrics.model_dump_json(indent=2)}\n\n"
        f"Suggested optimization themes:\n{hints_block}\n\n"
        f"Current kernel.cu:\n```cuda\n{src}\n```"
    )


def _extract_source_from_response(response: Any) -> str | None:
    src = _source_from_response(response.text, response.tool_calls)
    return src or None
