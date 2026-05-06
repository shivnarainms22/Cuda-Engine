from cuda_engine.config import SynthesisConfig


def test_default_retry_budgets() -> None:
    config = SynthesisConfig()

    assert config.retry_budgets.codegen == 3
    assert config.retry_budgets.correctness == 3
    assert config.retry_budgets.performance == 3
    assert config.retry_budgets.interview == 1
    assert config.retry_budgets.polish == 1


def test_perf_target_defaults() -> None:
    config = SynthesisConfig()

    assert config.perf_target_speedup_vs_torch_compile == 1.0
    assert config.escalate_to_opus_on_bust is True


def test_benchmark_defaults_are_large_enough_for_gpu_timing() -> None:
    config = SynthesisConfig()

    assert config.performance_shape_n == 1_048_576
    assert config.benchmark_warmup_iterations == 10
    assert config.benchmark_timed_iterations == 100
