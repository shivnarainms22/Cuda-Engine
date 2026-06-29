from pydantic import BaseModel, ConfigDict, Field

_SONNET_ROUTED = "anthropic:claude-sonnet-4-6"


class StageModels(BaseModel):
    model_config = ConfigDict(frozen=True)

    interview: str = _SONNET_ROUTED
    codegen: str = _SONNET_ROUTED
    correctness: str = _SONNET_ROUTED
    performance: str = _SONNET_ROUTED
    polish: str = _SONNET_ROUTED


class RetryBudgets(BaseModel):
    model_config = ConfigDict(frozen=True)

    interview: int = 1
    codegen: int = 3
    correctness: int = 3
    performance: int = 3
    polish: int = 1


class SynthesisConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    retry_budgets: RetryBudgets = Field(default_factory=RetryBudgets)
    escalate_to_opus_on_bust: bool = False
    perf_target_speedup_vs_torch_compile: float = 1.0
    correctness_rtol: float = 1e-3
    correctness_atol: float = 1e-3
    correctness_shapes: tuple[tuple[int, ...], ...] = ((0,), (1,), (127,), (128,), (1024,), (4097,))
    nvcc_flags: tuple[str, ...] = ("-O3", "--use_fast_math")
    artifact_root: str | None = None
    performance_shape_n: int = 16_777_216
    benchmark_warmup_iterations: int = 10
    benchmark_timed_iterations: int = 100
    sonnet_model: str = "claude-sonnet-4-6"
    opus_model: str = "claude-opus-4-7"
    opus_retry_budget_codegen: int = 1
    opus_retry_budget_performance: int = 1
    request_timeout_seconds: int = 120
    stage_models: StageModels = Field(default_factory=StageModels)
