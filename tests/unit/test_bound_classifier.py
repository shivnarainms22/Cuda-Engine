"""Tests for the perf bound classifier + bound-aware repair guidance."""
from __future__ import annotations

from cuda_engine.services.gpu.base import BenchmarkResult, NsightMetrics
from cuda_engine.stages.performance import (
    _bound_guidance,
    _format_perf_hints,
    classify_bound,
)

# Real signatures from the v1.0 run.
GELU_LATENCY = NsightMetrics(  # gelu_fp16 attempt_01
    memory_throughput_pct=35.78, compute_sm_pct=40.54, occupancy=0.4321, waves_per_sm=75.85,
    regs_per_thread=30, sol_bottleneck="...latency issues...",
)
RMS_BANDWIDTH = NsightMetrics(  # rms_norm_fp16 attempt_02
    memory_throughput_pct=86.34, compute_sm_pct=12.17, occupancy=0.863, waves_per_sm=4.74,
)
COMPUTE = NsightMetrics(memory_throughput_pct=30.0, compute_sm_pct=85.0, occupancy=0.7)
UNKNOWN = NsightMetrics()  # no signals (ncu unavailable)

_BENCH = BenchmarkResult(ok=True, custom_ms=1.0, baseline_ms=0.9)


def test_classify_latency_bound() -> None:
    assert classify_bound(GELU_LATENCY) == "latency"


def test_classify_bandwidth_bound() -> None:
    assert classify_bound(RMS_BANDWIDTH) == "bandwidth"


def test_classify_compute_bound() -> None:
    assert classify_bound(COMPUTE) == "compute"


def test_classify_unknown_when_signals_missing() -> None:
    assert classify_bound(UNKNOWN) == "unknown"


def test_latency_guidance_steers_away_from_adding_ilp() -> None:
    g = _bound_guidance(GELU_LATENCY)
    assert g is not None
    assert "latency-bound" in g
    assert "Do NOT add" in g
    assert "occupancy" in g


def test_bandwidth_guidance_says_move_less_memory() -> None:
    g = _bound_guidance(RMS_BANDWIDTH)
    assert g is not None
    assert "bandwidth-bound" in g
    assert "MOVE LESS MEMORY" in g
    assert "cache" in g


def test_unknown_metrics_yield_no_bound_guidance() -> None:
    assert _bound_guidance(UNKNOWN) is None


def test_format_perf_hints_puts_bound_guidance_first() -> None:
    hints = _format_perf_hints(GELU_LATENCY, benchmark=_BENCH)
    assert hints[0].startswith("MEASURED BOTTLENECK: latency-bound")


def test_format_perf_hints_without_signals_still_produces_hints() -> None:
    hints = _format_perf_hints(UNKNOWN, benchmark=_BENCH)
    assert hints  # falls back to the existing generic hints
    assert not hints[0].startswith("MEASURED BOTTLENECK")
