from pathlib import Path

from cuda_engine.models import (
    CorrectnessReport,
    KernelArtifact,
    KernelSpec,
    OptimizationPriority,
    PerformanceReport,
    PrecisionTolerance,
    TensorArg,
)
from cuda_engine.services.gpu.base import CompileResult, RunResult
from cuda_engine.services.gpu.mocks import MockGPURunner
from cuda_engine.services.llm.mocks import MockLLMClient
from cuda_engine.services.store.mocks import InMemoryStore
from cuda_engine.stages.polish import Stage5Polish

SHAPE_SIZES = (0, 1, 127, 128, 1024, 4097)


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


def test_stage5_polish_writes_annotated_kernel() -> None:
    torch = __import__("torch")
    kernel_path = Path(".test_artifacts/polish/kernel.cu")
    kernel_path.parent.mkdir(parents=True, exist_ok=True)
    kernel_path.write_text("__global__ void k() {}", encoding="utf-8")
    store = InMemoryStore()
    stage = Stage5Polish(
        llm=MockLLMClient(["```cuda\n// annotated tile choice\n__global__ void k() {}\n```"]),
        gpu=MockGPURunner(
            compile_results=[CompileResult(ok=True, so_path=Path("polished.so"), log="ok")],
            run_results=[
                RunResult(ok=True, output_tensors=[torch.arange(size, dtype=torch.float32)])
                for size in SHAPE_SIZES
            ],
        ),
        store=store,
    )

    artifact = stage.run(
        spec=_spec(),
        artifact=KernelArtifact(kernel_cu_path=kernel_path, kernel_so_path=Path("kernel.so")),
        correctness=CorrectnessReport(
            passed=True,
            max_abs_err=0.0,
            max_rel_err=0.0,
            shapes_tested=[(128,)],
        ),
        performance=PerformanceReport(speedup_vs_reference=1.0, speedup_vs_torch_compile=1.0),
        reference=lambda x, y: x,
        run_id="run123",
        model="claude-sonnet-4-6",
    )

    assert artifact.kernel_cu_path.as_posix().endswith("stage5_polish/final/kernel.cu")
    assert artifact.kernel_so_path == Path("polished.so")
    assert b"annotated tile choice" in store._files[("run123", "stage5_polish/kernel_annotated.cu")]
    assert b'"accepted": true' in store._files[("run123", "stage5_polish/status.json")]
    assert b'"passed": true' in store._files[("run123", "stage5_polish/correctness_report.json")]
    assert stage.llm is not None
    assert stage.llm.calls[0]["model"] == "claude-sonnet-4-6"


def test_stage5_polish_keeps_verified_kernel_when_annotation_does_not_compile() -> None:
    original = KernelArtifact(
        kernel_cu_path=Path(".test_artifacts/polish/original.cu"),
        kernel_so_path=Path("original.so"),
        compile_log="original ok",
    )
    original.kernel_cu_path.parent.mkdir(parents=True, exist_ok=True)
    original.kernel_cu_path.write_text("__global__ void original() {}", encoding="utf-8")
    store = InMemoryStore()
    stage = Stage5Polish(
        llm=MockLLMClient(["```cuda\nbroken annotated source\n```"]),
        gpu=MockGPURunner(
            compile_results=[
                CompileResult(ok=False, so_path=None, log="compile failed", errors=["syntax error"])
            ]
        ),
        store=store,
    )

    artifact = stage.run(
        spec=_spec(),
        artifact=original,
        correctness=CorrectnessReport(
            passed=True,
            max_abs_err=0.0,
            max_rel_err=0.0,
            shapes_tested=[(128,)],
        ),
        performance=PerformanceReport(speedup_vs_reference=1.0, speedup_vs_torch_compile=1.0),
        reference=lambda x, y: x,
        run_id="run123",
        model="claude-sonnet-4-6",
    )

    assert artifact == original
    assert b'"accepted": false' in store._files[("run123", "stage5_polish/status.json")]
    assert b"compile failed" in store._files[("run123", "stage5_polish/compile.log")]
