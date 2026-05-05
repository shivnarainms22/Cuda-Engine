from pathlib import Path

from cuda_engine.models import (
    KernelArtifact,
    KernelSpec,
    OptimizationPriority,
    PrecisionTolerance,
    TensorArg,
)
from cuda_engine.services.gpu.base import RunResult
from cuda_engine.services.gpu.mocks import MockGPURunner
from cuda_engine.services.store.mocks import InMemoryStore
from cuda_engine.stages.correctness import Stage3Correctness


def _identity_spec() -> KernelSpec:
    return KernelSpec(
        name="identity",
        target_arch="sm_80",
        inputs=[TensorArg(name="x", dtype="fp32", shape=("N",))],
        outputs=[TensorArg(name="out", dtype="fp32", shape=("N",))],
        precision_tolerance=PrecisionTolerance(rtol=1e-5, atol=1e-6),
        optimization_priority=OptimizationPriority.BALANCED,
    )


def test_stage3_correctness_passes_when_kernel_matches_reference() -> None:
    torch = __import__("torch")
    stage = Stage3Correctness(
        gpu=MockGPURunner(run_results=[RunResult(ok=True, output_tensors=[torch.arange(128.0)])]),
        store=InMemoryStore(),
    )

    report = stage.run(
        spec=_identity_spec(),
        artifact=KernelArtifact(kernel_cu_path=Path("kernel.cu"), kernel_so_path=Path("kernel.so")),
        reference=lambda x: x,
        run_id="run123",
    )

    assert report.passed is True
    assert report.max_abs_err == 0.0
    assert report.shapes_tested == [(128,)]


def test_stage3_correctness_fails_when_kernel_differs_from_reference() -> None:
    torch = __import__("torch")
    stage = Stage3Correctness(
        gpu=MockGPURunner(run_results=[RunResult(ok=True, output_tensors=[torch.zeros(128)])]),
        store=InMemoryStore(),
    )

    report = stage.run(
        spec=_identity_spec(),
        artifact=KernelArtifact(kernel_cu_path=Path("kernel.cu"), kernel_so_path=Path("kernel.so")),
        reference=lambda x: x,
        run_id="run123",
    )

    assert report.passed is False
    assert report.max_abs_err > 0
    assert report.failing_inputs[0]["shape"] == (128,)


def test_stage3_correctness_fails_when_kernel_run_fails() -> None:
    stage = Stage3Correctness(
        gpu=MockGPURunner(run_results=[RunResult(ok=False, stderr="launch failed")]),
        store=InMemoryStore(),
    )

    report = stage.run(
        spec=_identity_spec(),
        artifact=KernelArtifact(kernel_cu_path=Path("kernel.cu"), kernel_so_path=Path("kernel.so")),
        reference=lambda x: x,
        run_id="run123",
    )

    assert report.passed is False
    assert report.failing_inputs[0]["error"] == "launch failed"
