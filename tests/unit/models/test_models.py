from pathlib import Path

import pytest
from pydantic import ValidationError

from cuda_engine.models.artifact import KernelArtifact
from cuda_engine.models.reports import CorrectnessReport, SynthesisReport
from cuda_engine.models.spec import KernelSpec, OptimizationPriority, TensorArg


def test_kernel_spec_minimal_round_trip() -> None:
    spec = KernelSpec(
        name="vector_add",
        target_arch="sm_80",
        inputs=[
            TensorArg(name="x", dtype="fp32", shape=("N",)),
            TensorArg(name="y", dtype="fp32", shape=("N",)),
        ],
        outputs=[TensorArg(name="out", dtype="fp32", shape=("N",))],
        precision_tolerance={"rtol": 1e-5, "atol": 1e-6},
        optimization_priority=OptimizationPriority.THROUGHPUT,
    )

    parsed = KernelSpec.model_validate_json(spec.model_dump_json())

    assert parsed == spec


def test_kernel_spec_rejects_unknown_dtype() -> None:
    with pytest.raises(ValidationError):
        KernelSpec(
            name="bad",
            target_arch="sm_80",
            inputs=[TensorArg(name="x", dtype="float37", shape=("N",))],
            outputs=[TensorArg(name="o", dtype="fp32", shape=("N",))],
            precision_tolerance={"rtol": 1e-5, "atol": 1e-6},
            optimization_priority=OptimizationPriority.LATENCY,
        )


def test_kernel_artifact_round_trip() -> None:
    artifact = KernelArtifact(
        kernel_cu_path=Path("kernel.cu"),
        kernel_so_path=Path("kernel.so"),
        compile_log="ok",
        ptx_size_bytes=1234,
    )

    parsed = KernelArtifact.model_validate_json(artifact.model_dump_json())

    assert parsed.kernel_cu_path == artifact.kernel_cu_path
    assert parsed.kernel_so_path == artifact.kernel_so_path


def test_correctness_report_passed_property() -> None:
    report = CorrectnessReport(
        passed=True,
        max_abs_err=1e-6,
        max_rel_err=1e-6,
        shapes_tested=[(128,), (1024,)],
        failing_inputs=[],
    )

    assert report.passed is True


def test_synthesis_report_aggregates() -> None:
    report = SynthesisReport(
        run_id="abc123",
        spec_name="vector_add",
        stages_executed=["interview", "codegen", "correctness", "performance", "polish"],
        total_llm_tokens_in=1000,
        total_llm_tokens_out=400,
        total_cost_usd=0.05,
        wall_time_seconds=42.0,
    )

    assert report.run_id == "abc123"
