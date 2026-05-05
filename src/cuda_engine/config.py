from pydantic import BaseModel, ConfigDict, Field


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
    escalate_to_opus_on_bust: bool = True
    perf_target_speedup_vs_torch_compile: float = 1.0
    correctness_rtol: float = 1e-3
    correctness_atol: float = 1e-3
    nvcc_flags: tuple[str, ...] = ("-O3", "--use_fast_math")
    artifact_root: str | None = None
    sonnet_model: str = "claude-sonnet-4-6"
    opus_model: str = "claude-opus-4-7"
    request_timeout_seconds: int = 120
