from pathlib import Path

from cuda_engine.services.gpu.local import parse_ncu_csv

FIXTURE_PATH = Path(__file__).parents[3] / "fixtures" / "ncu_basic_vector_add.csv"


def test_parse_ncu_csv_extracts_occupancy_and_regs_from_basic_set() -> None:
    csv_text = FIXTURE_PATH.read_text(encoding="utf-8")

    metrics = parse_ncu_csv(csv_text)

    assert metrics.occupancy is not None
    assert abs(metrics.occupancy - 0.7448) < 1e-3
    assert metrics.regs_per_thread == 16
    assert metrics.uncoalesced_global_loads_pct is None
    assert metrics.spill_bytes == 0
    assert metrics.achieved_bandwidth_gbps is None
    assert metrics.achieved_tflops is None
    assert "Achieved Occupancy" in metrics.raw_csv


def test_parse_ncu_csv_skips_leading_non_csv_noise() -> None:
    csv_text = (
        "==PROF== Connected to process 1\n"
        "kernel ran successfully\n"
        "==PROF== Disconnected from process 1\n"
        '"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream",'
        '"Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit",'
        '"Metric Value","Rule Name","Rule Type","Rule Description",'
        '"Estimated Speedup Type","Estimated Speedup"\n'
        '"0","1","p","h","k","1","7","(256,1,1)","(4,1,1)","0","8.0","Occupancy",'
        '"Achieved Occupancy","%","50.00",'
        '"","","","",""\n'
        '"0","1","p","h","k","1","7","(256,1,1)","(4,1,1)","0","8.0","Launch Statistics",'
        '"Registers Per Thread","register/thread","32",'
        '"","","","",""\n'
    )

    metrics = parse_ncu_csv(csv_text)

    assert abs(metrics.occupancy - 0.5) < 1e-6
    assert metrics.regs_per_thread == 32


def test_parse_ncu_csv_returns_empty_metrics_when_csv_missing_target_metrics() -> None:
    csv_text = (
        '"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream",'
        '"Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit",'
        '"Metric Value","Rule Name","Rule Type","Rule Description",'
        '"Estimated Speedup Type","Estimated Speedup"\n'
    )

    metrics = parse_ncu_csv(csv_text)

    assert metrics.occupancy is None
    assert metrics.regs_per_thread is None
    assert metrics.raw_csv == csv_text


def test_parse_ncu_csv_handles_thousands_separator_in_metric_value() -> None:
    csv_text = (
        '"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream",'
        '"Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit",'
        '"Metric Value","Rule Name","Rule Type","Rule Description",'
        '"Estimated Speedup Type","Estimated Speedup"\n'
        '"0","1","p","h","k","1","7","(256,1,1)","(4,1,1)","0","8.0","Launch Statistics",'
        '"Registers Per Thread","register/thread","1,024",'
        '"","","","",""\n'
    )

    metrics = parse_ncu_csv(csv_text)

    assert metrics.regs_per_thread == 1024


def test_parse_ncu_csv_uses_first_pass_when_replays_present() -> None:
    csv_text = (
        '"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream",'
        '"Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit",'
        '"Metric Value","Rule Name","Rule Type","Rule Description",'
        '"Estimated Speedup Type","Estimated Speedup"\n'
        '"0","1","p","h","k","1","7","(256,1,1)","(4,1,1)","0","8.0","Occupancy",'
        '"Achieved Occupancy","%","60.00",'
        '"","","","",""\n'
        '"1","1","p","h","k","1","7","(256,1,1)","(4,1,1)","0","8.0","Occupancy",'
        '"Achieved Occupancy","%","99.00",'
        '"","","","",""\n'
    )

    metrics = parse_ncu_csv(csv_text)

    assert abs(metrics.occupancy - 0.60) < 1e-6
