from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field


class CorrectnessReport(BaseModel):
    passed: bool
    max_abs_err: float
    max_rel_err: float
    shapes_tested: list[tuple[int, ...]]
    failing_inputs: list[dict[str, Any]] = Field(default_factory=list)


class PerformanceReport(BaseModel):
    speedup_vs_reference: float
    speedup_vs_torch_compile: float
    achieved_tflops: float | None = None
    achieved_gbps: float | None = None
    occupancy: float | None = None
    regs_per_thread: int | None = None
    spill_bytes: int = 0
    below_target: bool = False
    notes: list[str] = Field(default_factory=list)


class StageTrace(BaseModel):
    stage_name: str
    attempts: int
    succeeded: bool
    model_used: str
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read_tokens: int = 0
    latency_seconds: float = 0.0


class SynthesisReport(BaseModel):
    run_id: str
    spec_name: str
    stages_executed: list[str]
    stage_traces: list[StageTrace] = Field(default_factory=list)
    total_llm_tokens_in: int = 0
    total_llm_tokens_out: int = 0
    total_cost_usd: float = 0.0
    wall_time_seconds: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class SynthesisResult(BaseModel):
    """Top-level return value of synthesize()."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    passed: bool
    run_id: str
    artifacts_dir: str
    report: SynthesisReport
    failed_stage: int | None = None
    failure_reason: str | None = None
    correctness: CorrectnessReport | None = None
    performance: PerformanceReport | None = None
    kernel_callable: object | None = None

    @classmethod
    def ok(
        cls,
        *,
        run_id: str,
        artifacts_dir: str,
        report: SynthesisReport,
        correctness: CorrectnessReport,
        performance: PerformanceReport,
        kernel_callable: object | None,
    ) -> Self:
        return cls(
            passed=True,
            run_id=run_id,
            artifacts_dir=artifacts_dir,
            report=report,
            correctness=correctness,
            performance=performance,
            kernel_callable=kernel_callable,
        )

    @classmethod
    def failed(
        cls,
        *,
        stage: int,
        reason: str,
        run_id: str,
        artifacts_dir: str,
        report: SynthesisReport,
        correctness: CorrectnessReport | None = None,
    ) -> Self:
        return cls(
            passed=False,
            failed_stage=stage,
            failure_reason=reason,
            run_id=run_id,
            artifacts_dir=artifacts_dir,
            report=report,
            correctness=correctness,
        )
