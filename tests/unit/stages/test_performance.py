from pathlib import Path

from cuda_engine.models import (
    KernelArtifact,
    KernelSpec,
    OptimizationPriority,
    PrecisionTolerance,
    TensorArg,
)
from cuda_engine.services.gpu.base import BenchmarkResult
from cuda_engine.services.gpu.mocks import MockGPURunner
from cuda_engine.services.store.mocks import InMemoryStore
from cuda_engine.stages.performance import Stage4Performance


def test_stage4_performance_uses_benchmark_result_and_writes_report() -> None:
    store = InMemoryStore()
    gpu = MockGPURunner(
        benchmark_results=[
            BenchmarkResult(
                ok=True,
                custom_ms=0.25,
                baseline_ms=1.0,
                achieved_gbps=512.0,
                warmup_iterations=5,
                timed_iterations=20,
            )
        ]
    )
    stage = Stage4Performance(gpu=gpu, store=store)

    report = stage.run(
        spec=_spec(),
        artifact=KernelArtifact(kernel_cu_path=Path("kernel.cu"), kernel_so_path=Path("kernel.so")),
        run_id="run123",
    )

    assert report.speedup_vs_reference == 4.0
    assert report.speedup_vs_torch_compile == 4.0
    assert report.achieved_gbps == 512.0
    assert report.below_target is False
    assert b'"speedup_vs_reference": 4.0' in store._files[("run123", "stage4_performance/report.json")]


def test_stage4_performance_reports_missing_shared_object() -> None:
    store = InMemoryStore()
    stage = Stage4Performance(gpu=MockGPURunner(), store=store)

    report = stage.run(
        spec=_spec(),
        artifact=KernelArtifact(kernel_cu_path=Path("kernel.cu"), kernel_so_path=None),
        run_id="run123",
    )

    assert report.speedup_vs_reference == 0.0
    assert report.speedup_vs_torch_compile == 0.0
    assert report.below_target is True
    assert b"kernel_so_path is required" in store._files[("run123", "stage4_performance/report.json")]


def _spec() -> KernelSpec:
    return KernelSpec(
        name="vector_add",
        target_arch="sm_80",
        inputs=[
            TensorArg(name="x", dtype="fp32", shape=("N",)),
            TensorArg(name="y", dtype="fp32", shape=("N",)),
        ],
        outputs=[TensorArg(name="out", dtype="fp32", shape=("N",))],
        precision_tolerance=PrecisionTolerance(rtol=1e-5, atol=1e-6),
        optimization_priority=OptimizationPriority.THROUGHPUT,
    )
