"""Tests for ncu-CSV bottleneck-signal extraction (parse_ncu_csv).

The metric values below are REAL measurements captured from the v1.0 run
`2026-06-01-191749` (gelu_fp16 / rms_norm_fp16 perf-repair attempts). They are
the signals the Stage-4 repair loop needs to tell latency- from bandwidth-bound.
"""
from __future__ import annotations

import csv
import io

from cuda_engine.services.gpu.local import parse_ncu_csv

_HEADER = [
    "ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context",
    "Stream", "Block Size", "Grid Size", "Device", "CC", "Section Name",
    "Metric Name", "Metric Unit", "Metric Value", "Rule Name", "Rule Type",
    "Rule Description", "Estimated Speedup Type", "Estimated Speedup",
]


def _csv(rows: list[dict[str, str]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_HEADER, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    for row in rows:
        full = {col: "" for col in _HEADER}
        full["ID"] = "0"
        full["Process ID"] = "43740"
        full.update(row)
        writer.writerow(full)
    return buf.getvalue()


def _m(section: str, name: str, unit: str, value: str) -> dict[str, str]:
    return {"Section Name": section, "Metric Name": name, "Metric Unit": unit, "Metric Value": value}


_SOL = "GPU Speed Of Light Throughput"

# gelu_fp16 perf_repair attempt_01 — latency-bound (low memory AND compute %)
GELU_A1 = _csv([
    _m(_SOL, "Memory Throughput", "%", "35.78"),
    _m(_SOL, "DRAM Throughput", "%", "35.78"),
    _m(_SOL, "Compute (SM) Throughput", "%", "40.54"),
    {
        "Section Name": "SpeedOfLight",
        "Rule Name": "SOLBottleneck",
        "Rule Type": "OPT",
        "Rule Description": (
            "This workload exhibits low compute throughput and memory bandwidth "
            "utilization relative to the peak performance of this device. "
            "Achieved compute throughput and/or memory bandwidth below 60.0% of "
            "peak typically indicate latency issues."
        ),
    },
    _m("Launch Statistics", "Registers Per Thread", "register/thread", "30"),
    _m("Launch Statistics", "Waves Per SM", "", "75.85"),
    _m("Occupancy", "Achieved Occupancy", "%", "43.21"),
])

# rms_norm_fp16 perf_repair attempt_02 — bandwidth-bound (DRAM 86%)
RMS_A2 = _csv([
    _m(_SOL, "Memory Throughput", "%", "86.34"),
    _m(_SOL, "DRAM Throughput", "%", "86.34"),
    _m(_SOL, "Compute (SM) Throughput", "%", "12.17"),
    {
        "Section Name": "SpeedOfLight",
        "Rule Name": "SOLBottleneck",
        "Rule Type": "INF",
        "Rule Description": (
            "This workload is utilizing greater than 80.0% of the available "
            "compute or memory performance of the device."
        ),
    },
    _m("Launch Statistics", "Registers Per Thread", "register/thread", "18"),
    _m("Launch Statistics", "Waves Per SM", "", "4.74"),
    _m("Occupancy", "Achieved Occupancy", "%", "86.30"),
])


def test_extracts_latency_bound_signals() -> None:
    m = parse_ncu_csv(GELU_A1)
    assert m.memory_throughput_pct == 35.78
    assert m.dram_throughput_pct == 35.78
    assert m.compute_sm_pct == 40.54
    assert m.waves_per_sm == 75.85
    assert m.occupancy is not None and abs(m.occupancy - 0.4321) < 1e-6
    assert m.regs_per_thread == 30
    assert "latency issues" in m.sol_bottleneck


def test_extracts_bandwidth_bound_signals() -> None:
    m = parse_ncu_csv(RMS_A2)
    assert m.memory_throughput_pct == 86.34
    assert m.compute_sm_pct == 12.17
    assert m.waves_per_sm == 4.74
    assert "80.0%" in m.sol_bottleneck


def test_missing_signals_stay_none() -> None:
    m = parse_ncu_csv("ncu_not_available")
    assert m.memory_throughput_pct is None
    assert m.waves_per_sm is None
    assert m.sol_bottleneck == ""
