from pathlib import Path

from cuda_engine.config import SynthesisConfig
from cuda_engine.models import (
    KernelArtifact,
    KernelSpec,
    OptimizationPriority,
    PrecisionTolerance,
    TensorArg,
)
from cuda_engine.services.gpu.base import BenchmarkResult, CompileResult, NsightMetrics
from cuda_engine.services.gpu.mocks import MockGPURunner
from cuda_engine.services.llm.base import LLMResponse
from cuda_engine.services.llm.mocks import MockLLMClient
from cuda_engine.services.store.mocks import InMemoryStore
from cuda_engine.stages.performance import Stage4Performance, _format_perf_hints


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

    report, returned_artifact = stage.run(
        spec=_spec(),
        artifact=KernelArtifact(kernel_cu_path=Path("kernel.cu"), kernel_so_path=Path("kernel.so")),
        run_id="run123",
    )

    assert report.speedup_vs_reference == 4.0
    assert report.speedup_vs_torch_compile == 4.0
    assert report.achieved_gbps == 512.0
    assert report.below_target is False
    assert returned_artifact.kernel_so_path == Path("kernel.so")
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

    report, _ = stage.run(
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


def _initial_artifact_in_store(store: InMemoryStore, run_id: str, src: str) -> KernelArtifact:
    cu = store.write_text(run_id, "stage2_codegen/final/kernel.cu", src)
    so = store.write_bytes(run_id, "stage2_codegen/final/kernel.so", b"initial")
    return KernelArtifact(kernel_cu_path=cu, kernel_so_path=so, compile_log="ok")


def _llm_compile_response(src: str) -> LLMResponse:
    return LLMResponse(
        text=f"```cuda\n{src}\n```",
        model="claude-sonnet-4-6",
        tool_calls=[
            {"name": "compile_kernel", "input": {"src": src, "target_arch": "sm_80"}}
        ],
        tokens_in=100,
        tokens_out=200,
    )


def test_stage4_performance_retries_until_target_met() -> None:
    store = InMemoryStore()
    artifact = _initial_artifact_in_store(store, "run123", "// initial slow kernel")
    gpu = MockGPURunner(
        compile_results=[
            CompileResult(ok=True, so_path=Path("/tmp/v2.so"), log="ok", ptx_size_bytes=1024),
        ],
        benchmark_results=[
            BenchmarkResult(
                ok=True, custom_ms=4.0, baseline_ms=2.0,
                warmup_iterations=10, timed_iterations=50,
            ),
            BenchmarkResult(
                ok=True, custom_ms=1.0, baseline_ms=2.0, achieved_gbps=300.0,
                warmup_iterations=10, timed_iterations=50,
            ),
        ],
        profile_results=[NsightMetrics(occupancy=0.45, regs_per_thread=80)],
    )
    llm = MockLLMClient([_llm_compile_response("// faster kernel")])
    cfg = SynthesisConfig(perf_target_speedup_vs_torch_compile=1.0)
    stage = Stage4Performance(llm=llm, gpu=gpu, store=store, cfg=cfg)

    report, final_artifact = stage.run(
        spec=_spec(), artifact=artifact, run_id="run123", retry_budget=3
    )

    assert report.below_target is False
    assert report.speedup_vs_torch_compile == 2.0
    assert report.warnings == []
    assert any("attempt 1: speedup 0.500 -> 2.000" in note for note in report.notes)
    assert final_artifact.kernel_cu_path != artifact.kernel_cu_path
    assert llm.call_count == 1


def test_stage4_performance_soft_fails_after_exhausted_retry_budget() -> None:
    store = InMemoryStore()
    artifact = _initial_artifact_in_store(store, "run123", "// initial")
    benchmark_below = BenchmarkResult(
        ok=True, custom_ms=4.0, baseline_ms=2.0, warmup_iterations=10, timed_iterations=50
    )
    gpu = MockGPURunner(
        compile_results=[
            CompileResult(ok=True, so_path=Path("/tmp/v2.so"), log="ok"),
            CompileResult(ok=True, so_path=Path("/tmp/v3.so"), log="ok"),
        ],
        benchmark_results=[benchmark_below, benchmark_below, benchmark_below],
        profile_results=[
            NsightMetrics(occupancy=0.4, regs_per_thread=72),
            NsightMetrics(occupancy=0.4, regs_per_thread=72),
        ],
    )
    llm = MockLLMClient(
        [
            _llm_compile_response("// candidate v2"),
            _llm_compile_response("// candidate v3"),
        ]
    )
    cfg = SynthesisConfig(perf_target_speedup_vs_torch_compile=1.0, opus_retry_budget_performance=0)
    stage = Stage4Performance(llm=llm, gpu=gpu, store=store, cfg=cfg)

    report, _ = stage.run(spec=_spec(), artifact=artifact, run_id="run123", retry_budget=2)

    assert report.below_target is True
    assert report.speedup_vs_torch_compile == 0.5
    assert any("budget exhausted" in w for w in report.warnings)


def test_stage4_performance_records_failed_compile_in_warnings_and_continues() -> None:
    store = InMemoryStore()
    artifact = _initial_artifact_in_store(store, "run123", "// initial")
    gpu = MockGPURunner(
        compile_results=[
            CompileResult(ok=False, log="compile failed", errors=["nvcc: error"]),
            CompileResult(ok=True, so_path=Path("/tmp/v2.so"), log="ok"),
        ],
        benchmark_results=[
            BenchmarkResult(ok=True, custom_ms=4.0, baseline_ms=2.0, warmup_iterations=10, timed_iterations=50),
            BenchmarkResult(ok=True, custom_ms=1.0, baseline_ms=2.0, warmup_iterations=10, timed_iterations=50),
        ],
        profile_results=[
            NsightMetrics(occupancy=0.4, regs_per_thread=72),
            NsightMetrics(occupancy=0.4, regs_per_thread=72),
        ],
    )
    llm = MockLLMClient(
        [
            _llm_compile_response("// bad candidate"),
            _llm_compile_response("// good candidate"),
        ]
    )
    cfg = SynthesisConfig(perf_target_speedup_vs_torch_compile=1.0)
    stage = Stage4Performance(llm=llm, gpu=gpu, store=store, cfg=cfg)

    report, _ = stage.run(spec=_spec(), artifact=artifact, run_id="run123", retry_budget=3)

    assert report.below_target is False
    assert report.speedup_vs_torch_compile == 2.0
    assert any("attempt 1: compile failed" in w for w in report.warnings)


def test_stage4_performance_skips_retry_when_initial_speedup_meets_target() -> None:
    store = InMemoryStore()
    artifact = _initial_artifact_in_store(store, "run123", "// initial")
    gpu = MockGPURunner(
        benchmark_results=[
            BenchmarkResult(ok=True, custom_ms=0.5, baseline_ms=2.0, warmup_iterations=10, timed_iterations=50),
        ],
    )
    llm = MockLLMClient([])
    cfg = SynthesisConfig(perf_target_speedup_vs_torch_compile=1.0)
    stage = Stage4Performance(llm=llm, gpu=gpu, store=store, cfg=cfg)

    report, returned = stage.run(spec=_spec(), artifact=artifact, run_id="run123", retry_budget=3)

    assert report.below_target is False
    assert report.speedup_vs_torch_compile == 4.0
    assert llm.call_count == 0
    assert returned is artifact


def test_format_perf_hints_flags_register_pressure_and_low_occupancy() -> None:
    metrics = NsightMetrics(occupancy=0.4, regs_per_thread=80, spill_bytes=64)
    benchmark = BenchmarkResult(
        ok=True, custom_ms=4.0, baseline_ms=2.0, warmup_iterations=10, timed_iterations=50
    )

    hints = _format_perf_hints(metrics, benchmark=benchmark)

    assert any("Register pressure" in h for h in hints)
    assert any("occupancy" in h.lower() for h in hints)
    assert any("Spill bytes" in h for h in hints)
    assert any("slower than the eager baseline" in h for h in hints)


def test_format_perf_hints_falls_back_when_no_metrics_flag_anything() -> None:
    metrics = NsightMetrics(occupancy=0.9, regs_per_thread=32)
    benchmark = BenchmarkResult(
        ok=True, custom_ms=1.0, baseline_ms=2.0, warmup_iterations=10, timed_iterations=50
    )

    hints = _format_perf_hints(metrics, benchmark=benchmark)

    assert len(hints) == 1
    assert "No specific bottleneck" in hints[0]


def test_perf_retry_loop_uses_model_and_offset(tmp_path: Path) -> None:
    """_retry_loop should send the given model to LLM and number attempts after the offset."""
    store = InMemoryStore()
    artifact = _initial_artifact_in_store(store, "run123", "// initial slow kernel")
    benchmark_below = BenchmarkResult(
        ok=True, custom_ms=4.0, baseline_ms=2.0, warmup_iterations=10, timed_iterations=50
    )
    benchmark_above = BenchmarkResult(
        ok=True, custom_ms=0.5, baseline_ms=2.0, achieved_gbps=300.0,
        warmup_iterations=10, timed_iterations=50,
    )
    gpu = MockGPURunner(
        compile_results=[
            CompileResult(ok=True, so_path=Path("/tmp/v2.so"), log="ok", ptx_size_bytes=1024),
        ],
        benchmark_results=[benchmark_above],
        profile_results=[NsightMetrics(occupancy=0.45, regs_per_thread=80)],
    )
    llm = MockLLMClient([_llm_compile_response("// faster kernel")])
    cfg = SynthesisConfig(
        perf_target_speedup_vs_torch_compile=1.0,
        performance_shape_n=256,
        benchmark_warmup_iterations=2,
        benchmark_timed_iterations=3,
    )
    stage = Stage4Performance(llm=llm, gpu=gpu, store=store, cfg=cfg)

    stage._retry_loop(
        spec=_spec(),
        artifact=artifact,
        benchmark=benchmark_below,
        speedup=0.5,
        target=1.0,
        inputs=[],
        run_id="run123",
        retry_budget=1,
        model="claude-opus-4-7",
        attempt_offset=3,
    )

    # Model forwarded to LLM
    assert llm.calls[-1]["model"] == "claude-opus-4-7"
    # Artifact stored under attempt_04/ (offset 3 + local_attempt 1 = 4)
    assert ("run123", "stage4_performance/perf_repair/attempt_04/kernel.cu") in store._files


def test_stage4_escalates_to_opus_when_below_target() -> None:
    """When Sonnet loop ends below target, Stage 4 runs additional Opus iterations."""
    store = InMemoryStore()
    artifact = _initial_artifact_in_store(store, "run123", "// initial slow kernel")
    benchmark_below = BenchmarkResult(
        ok=True, custom_ms=4.0, baseline_ms=2.0, warmup_iterations=10, timed_iterations=50
    )
    benchmark_above = BenchmarkResult(
        ok=True, custom_ms=0.5, baseline_ms=2.0, achieved_gbps=400.0,
        warmup_iterations=10, timed_iterations=50,
    )
    # initial benchmark (below) + sonnet-repair benchmark (below) + opus-repair benchmark (above)
    gpu = MockGPURunner(
        compile_results=[
            CompileResult(ok=True, so_path=Path("/tmp/v2.so"), log="ok"),
            CompileResult(ok=True, so_path=Path("/tmp/v3.so"), log="ok"),
        ],
        benchmark_results=[
            benchmark_below,   # initial benchmark → triggers retry loop
            benchmark_below,   # sonnet attempt_01 after-recompile → still below
            benchmark_above,   # opus attempt_02 after-recompile → above target
        ],
        profile_results=[
            NsightMetrics(occupancy=0.4, regs_per_thread=72),
            NsightMetrics(occupancy=0.4, regs_per_thread=72),
        ],
    )
    llm = MockLLMClient([
        LLMResponse(
            text="// sonnet kernel",
            model="claude-sonnet-4-6",
            tool_calls=[{"name": "compile_kernel", "input": {"src": "// sonnet kernel", "target_arch": "sm_80"}}],
        ),
        LLMResponse(
            text="// opus kernel",
            model="claude-opus-4-7",
            tool_calls=[{"name": "compile_kernel", "input": {"src": "// opus kernel", "target_arch": "sm_80"}}],
        ),
    ])
    cfg = SynthesisConfig(
        perf_target_speedup_vs_torch_compile=1.0,
        opus_retry_budget_performance=1,
        escalate_to_opus_on_bust=True,
        performance_shape_n=256,
        benchmark_warmup_iterations=2,
        benchmark_timed_iterations=3,
    )
    stage = Stage4Performance(llm=llm, gpu=gpu, store=store, cfg=cfg)

    report, _ = stage.run(spec=_spec(), artifact=artifact, run_id="run123", retry_budget=1)

    # Two LLM calls: one sonnet, one opus
    assert llm.call_count == 2
    assert llm.calls[0]["model"] == "claude-sonnet-4-6"
    assert llm.calls[1]["model"] == "claude-opus-4-7"
    # Opus succeeded → below_target is False
    assert report.below_target is False
    # Escalation note present
    assert any("escalated to opus" in note for note in report.notes)
    # Both attempt directories exist in store
    assert ("run123", "stage4_performance/perf_repair/attempt_01/kernel.cu") in store._files
    assert ("run123", "stage4_performance/perf_repair/attempt_02/kernel.cu") in store._files


def test_stage4_skips_escalation_when_disabled() -> None:
    """opus_retry_budget_performance=0 → no Opus loop, even when below_target."""
    store = InMemoryStore()
    artifact = _initial_artifact_in_store(store, "run123", "// initial slow kernel")
    benchmark_below = BenchmarkResult(
        ok=True, custom_ms=4.0, baseline_ms=2.0, warmup_iterations=10, timed_iterations=50
    )
    gpu = MockGPURunner(
        compile_results=[
            CompileResult(ok=True, so_path=Path("/tmp/v2.so"), log="ok"),
        ],
        benchmark_results=[
            benchmark_below,   # initial benchmark
            benchmark_below,   # sonnet attempt_01 after-recompile → still below
        ],
        profile_results=[NsightMetrics(occupancy=0.4, regs_per_thread=72)],
    )
    llm = MockLLMClient([
        LLMResponse(
            text="// sonnet kernel",
            model="claude-sonnet-4-6",
            tool_calls=[{"name": "compile_kernel", "input": {"src": "// sonnet kernel", "target_arch": "sm_80"}}],
        ),
    ])
    cfg = SynthesisConfig(
        perf_target_speedup_vs_torch_compile=1.0,
        opus_retry_budget_performance=0,
        escalate_to_opus_on_bust=True,
        performance_shape_n=256,
        benchmark_warmup_iterations=2,
        benchmark_timed_iterations=3,
    )
    stage = Stage4Performance(llm=llm, gpu=gpu, store=store, cfg=cfg)

    report, _ = stage.run(spec=_spec(), artifact=artifact, run_id="run123", retry_budget=1)

    # Only 1 LLM call (Sonnet only)
    assert llm.call_count == 1
    # Still below_target since Opus never ran
    assert report.below_target is True
