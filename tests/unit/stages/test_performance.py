from pathlib import Path

from cuda_engine.config import SynthesisConfig
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
                warmup_iterations=10,
                timed_iterations=100,
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
    assert b'"warmup_iterations": 10' in store._files[("run123", "stage4_performance/benchmark.json")]
    assert b'"timed_iterations": 100' in store._files[("run123", "stage4_performance/benchmark.json")]
    assert b'"performance_shape_n": 1048576' in store._files[
        ("run123", "stage4_performance/benchmark.json")
    ]


def test_stage4_performance_uses_configured_benchmark_settings() -> None:
    store = InMemoryStore()
    gpu = MockGPURunner(
        benchmark_results=[
            BenchmarkResult(
                ok=True,
                custom_ms=1.0,
                baseline_ms=1.0,
                warmup_iterations=2,
                timed_iterations=3,
            )
        ]
    )
    stage = Stage4Performance(
        gpu=gpu,
        store=store,
        cfg=SynthesisConfig(
            performance_shape_n=256,
            benchmark_warmup_iterations=2,
            benchmark_timed_iterations=3,
        ),
    )

    stage.run(
        spec=_spec(),
        artifact=KernelArtifact(kernel_cu_path=Path("kernel.cu"), kernel_so_path=Path("kernel.so")),
        run_id="run123",
    )

    assert gpu.benchmark_calls == [
        {
            "so_path": Path("kernel.so"),
            "input_shapes": [(256,), (256,)],
            "warmup_iterations": 2,
            "timed_iterations": 3,
            "timeout_seconds": 60,
        }
    ]


def test_stage4_performance_derives_rank_aware_benchmark_shape() -> None:
    store = InMemoryStore()
    gpu = MockGPURunner()
    stage = Stage4Performance(
        gpu=gpu,
        store=store,
        cfg=SynthesisConfig(performance_shape_n=16),
    )

    stage.run(
        spec=_matrix_spec(),
        artifact=KernelArtifact(kernel_cu_path=Path("kernel.cu"), kernel_so_path=Path("kernel.so")),
        run_id="run123",
    )

    assert gpu.benchmark_calls[0]["input_shapes"] == [(4, 4)]


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


def _matrix_spec() -> KernelSpec:
    return KernelSpec(
        name="matrix_identity",
        target_arch="sm_80",
        inputs=[TensorArg(name="x", dtype="fp32", shape=("B", "D"))],
        outputs=[TensorArg(name="out", dtype="fp32", shape=("B", "D"))],
        precision_tolerance=PrecisionTolerance(rtol=1e-5, atol=1e-6),
        optimization_priority=OptimizationPriority.THROUGHPUT,
    )
