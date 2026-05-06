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
from cuda_engine.stages import correctness
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
        gpu=MockGPURunner(
            run_results=[
                RunResult(ok=True, output_tensors=[torch.arange(size, dtype=torch.float32)])
                for size in (0, 1, 127, 128, 1024, 4097)
            ]
        ),
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
    assert report.shapes_tested == [(0,), (1,), (127,), (128,), (1024,), (4097,)]
    assert [shape["shape"] for shape in report.shape_results] == [
        (0,),
        (1,),
        (127,),
        (128,),
        (1024,),
        (4097,),
    ]
    assert all(shape["passed"] for shape in report.shape_results)


def test_stage3_correctness_fails_when_kernel_differs_from_reference() -> None:
    torch = __import__("torch")
    run_results = [
        RunResult(ok=True, output_tensors=[torch.arange(size, dtype=torch.float32)])
        for size in (0, 1, 127)
    ]
    run_results.append(RunResult(ok=True, output_tensors=[torch.zeros(128)]))
    run_results.extend(
        RunResult(ok=True, output_tensors=[torch.arange(size, dtype=torch.float32)])
        for size in (1024, 4097)
    )
    stage = Stage3Correctness(
        gpu=MockGPURunner(run_results=run_results),
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
    assert report.shape_results[3]["shape"] == (128,)
    assert report.shape_results[3]["passed"] is False


def test_stage3_correctness_fails_when_kernel_run_fails() -> None:
    stage = Stage3Correctness(
        gpu=MockGPURunner(
            run_results=[
                RunResult(ok=True, output_tensors=[__import__("torch").arange(0.0)]),
                RunResult(ok=True, output_tensors=[__import__("torch").arange(1.0)]),
                RunResult(ok=False, stderr="launch failed"),
            ]
        ),
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
    assert report.failing_inputs[0]["shape"] == (127,)
    assert report.shape_results[2]["passed"] is False


def test_stage3_input_generation_uses_cuda_when_available(monkeypatch) -> None:
    class FakeTensor:
        def __init__(self, calls: list[tuple[str, object]]) -> None:
            self.calls = calls

        def reshape(self, shape):
            self.calls.append(("reshape", shape))
            return self

        def to(self, **kwargs):
            self.calls.append(("to", kwargs))
            return self

        def __add__(self, other):
            self.calls.append(("add", other))
            return self

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakeTorch:
        float32 = "float32"
        cuda = FakeCuda()

        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def arange(self, n, dtype):
            self.calls.append(("arange", {"n": n, "dtype": dtype}))
            return FakeTensor(self.calls)

    fake_torch = FakeTorch()
    monkeypatch.setattr(correctness, "_torch", lambda: fake_torch)

    correctness._make_inputs(_identity_spec(), shape=(128,))

    assert ("to", {"dtype": "float32", "device": "cuda"}) in fake_torch.calls
