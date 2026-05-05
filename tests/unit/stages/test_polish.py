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
from cuda_engine.services.llm.mocks import MockLLMClient
from cuda_engine.services.store.mocks import InMemoryStore
from cuda_engine.stages.polish import Stage5Polish


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
    kernel_path = Path(".test_artifacts/polish/kernel.cu")
    kernel_path.parent.mkdir(parents=True, exist_ok=True)
    kernel_path.write_text("__global__ void k() {}", encoding="utf-8")
    store = InMemoryStore()
    stage = Stage5Polish(
        llm=MockLLMClient(["```cuda\n// annotated tile choice\n__global__ void k() {}\n```"]),
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
        run_id="run123",
    )

    assert artifact.kernel_cu_path.as_posix().endswith("stage5_polish/kernel_annotated.cu")
    assert b"annotated tile choice" in store._files[("run123", "stage5_polish/kernel_annotated.cu")]
    assert stage.llm is not None
    assert stage.llm.calls[0]["model"] == "claude-sonnet-4-6"
