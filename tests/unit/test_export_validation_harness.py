"""The export-validation harness must not be able to report a false PASS.

`tools/export_validation/standalone_check.py` runs on a GPU, but its comparison
and input-generation helpers are pure and testable here. A harness that always
says PASS is worse than no harness -- cf. the WMMA guidance check, whose value
came entirely from having a negative control.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

_CHECK_PATH = Path(__file__).resolve().parents[2] / "tools" / "export_validation" / "standalone_check.py"


def _harness() -> Any:
    spec = importlib.util.spec_from_file_location("_standalone_check", _CHECK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_harness_module_loads() -> None:
    assert callable(_harness().main)


def test_symbolic_dims_bind_to_the_requested_size() -> None:
    h = _harness()
    decls = [{"shape": ["B", "D"]}, {"shape": ["D"]}]
    assert h._bind_shapes(decls, 32) == [(32, 32), (32,)]


def test_literal_dims_are_honoured_not_overwritten() -> None:
    h = _harness()
    assert h._bind_shapes([{"shape": ["B", "128"]}], 32) == [(32, 128)]


def test_scalar_shape_binds_to_empty_tuple() -> None:
    h = _harness()
    assert h._bind_shapes([{"shape": []}], 32) == [()]


def test_inputs_match_declared_shapes_and_dtypes() -> None:
    h = _harness()
    spec = {
        "inputs": [
            {"name": "x", "dtype": "fp16", "shape": ["B", "D"]},
            {"name": "n", "dtype": "int32", "shape": ["D"]},
        ]
    }
    inputs = h._make_inputs(torch, spec, 8, "cpu")
    assert [tuple(t.shape) for t in inputs] == [(8, 8), (8,)]
    assert inputs[0].dtype == torch.float16
    assert inputs[1].dtype == torch.int32


def test_matches_accepts_an_identical_result() -> None:
    h = _harness()
    a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert h._matches(torch, a, a.clone(), rtol=1e-3, atol=1e-3)


def test_matches_rejects_a_wrong_result() -> None:
    """The control the on-GPU run depends on: the comparison must be able to fail."""
    h = _harness()
    a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert not h._matches(torch, a, a + 1.0, rtol=1e-3, atol=1e-3)


def test_matches_rejects_a_shape_mismatch() -> None:
    h = _harness()
    a = torch.zeros(3, 4)
    assert not h._matches(torch, a, torch.zeros(4, 3), rtol=1e-3, atol=1e-3)


def test_matches_rejects_a_differing_output_count() -> None:
    h = _harness()
    a = torch.zeros(2)
    assert not h._matches(torch, [a, a], [a], rtol=1e-3, atol=1e-3)


def test_matches_handles_multiple_outputs() -> None:
    h = _harness()
    a, b = torch.zeros(2), torch.ones(3)
    assert h._matches(torch, (a, b), (a.clone(), b.clone()), rtol=1e-3, atol=1e-3)


def test_clean_process_guard_rejects_an_already_imported_generator() -> None:
    import sys

    import cuda_engine  # noqa: F401 - importing it IS the precondition under test

    h = _harness()
    assert "cuda_engine" in sys.modules
    with pytest.raises(h.CheckFailed, match="cuda_engine"):
        h._assert_clean_process()


def test_clean_process_guard_passes_when_the_generator_is_absent(monkeypatch: Any) -> None:
    import sys

    h = _harness()
    monkeypatch.delitem(sys.modules, "cuda_engine", raising=False)
    h._assert_clean_process()  # must not raise


# --- the control must work at GEMM magnitudes, not just near 1.0 -------------


def test_perturbation_is_detected_at_gemm_magnitudes() -> None:
    """Regression: a +1.0 nudge is invisible when outputs are ~1e11 and rtol=1e-3.

    Real numbers from the matmul_fp32 run: outputs ~1e11, so the tolerance band is
    rtol*|expected| ~1e8. The control must scale with the value.
    """
    h = _harness()
    expected = torch.full((4, 4), 2.0e11, dtype=torch.float32)
    actual = expected.clone()
    bad = h._perturb(torch, expected, rtol=1e-3, atol=1e-3)
    assert not h._matches(torch, actual, bad, rtol=1e-3, atol=1e-3)


def test_old_additive_perturbation_would_have_been_missed() -> None:
    """Pins the exact defect the A100 control caught, so it cannot come back."""
    h = _harness()
    expected = torch.full((4, 4), 2.0e11, dtype=torch.float32)
    naive_bad = expected + 1.0
    assert h._matches(torch, expected, naive_bad, rtol=1e-3, atol=1e-3), (
        "premise: +1.0 is inside the band at this magnitude"
    )


def test_perturbation_is_detected_near_unit_magnitude() -> None:
    h = _harness()
    expected = torch.tensor([[0.5, -1.25]], dtype=torch.float16)
    bad = h._perturb(torch, expected, rtol=1e-3, atol=1e-3)
    assert not h._matches(torch, expected, bad, rtol=1e-3, atol=1e-3)


def test_perturbation_is_detected_for_all_zero_output() -> None:
    """Scaling alone cannot perturb zeros; an absolute floor is required."""
    h = _harness()
    expected = torch.zeros(3, 3)
    bad = h._perturb(torch, expected, rtol=1e-3, atol=1e-3)
    assert not h._matches(torch, expected, bad, rtol=1e-3, atol=1e-3)


def test_perturbation_is_detected_for_integer_outputs() -> None:
    h = _harness()
    expected = torch.arange(6, dtype=torch.int64)
    bad = h._perturb(torch, expected, rtol=1e-3, atol=1e-3)
    assert not h._matches(torch, expected, bad, rtol=1e-3, atol=1e-3)


def test_error_stats_report_relative_not_just_absolute() -> None:
    h = _harness()
    expected = torch.full((2, 2), 1.0e12, dtype=torch.float32)
    actual = expected * 1.000001
    max_abs, max_rel = h._err_stats(torch, actual, expected)
    assert max_abs > 1e5, "absolute error is large and on its own meaningless"
    assert max_rel < 1e-4, "relative error is the number that matters"


# --- perf re-measurement ----------------------------------------------------


def test_timing_returns_a_median_not_a_mean(monkeypatch: Any) -> None:
    """One slow outlier must not dominate the reported time."""
    h = _harness()
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    delays = iter([0, 0, 0, 0, 0, 0.02, 0, 0, 0, 0, 0])
    import time as _time

    def call() -> None:
        d = next(delays, 0)
        if d:
            _time.sleep(d)

    ms = h._time_ms(torch, call, warmup=5, iters=5)
    assert ms < 5.0, "median should ignore the single 20ms outlier"


def test_regression_factor_is_documented_and_generous() -> None:
    h = _harness()
    assert 1.1 <= h._PERF_REGRESSION_FACTOR <= 1.5


def test_perf_args_are_skipped_when_the_run_has_no_benchmark(tmp_path: Path) -> None:
    import importlib.util

    path = _CHECK_PATH.parent / "validate_export.py"
    spec = importlib.util.spec_from_file_location("_validate_export", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._perf_args(tmp_path) == []


def test_perf_args_use_the_engines_own_benchmark_shape(tmp_path: Path) -> None:
    """Rank-2 spec at 16M elements must yield 4096, the engine's rule -- not a guess."""
    import importlib.util
    import json

    (tmp_path / "stage4_performance").mkdir(parents=True)
    (tmp_path / "stage4_performance" / "benchmark.json").write_text(
        json.dumps({
            "custom_ms": 1.5,
            "settings": {
                "performance_shape_n": 16_777_216,
                "benchmark_warmup_iterations": 10,
                "benchmark_timed_iterations": 100,
            },
        })
    )
    (tmp_path / "checkpoint.json").write_text(json.dumps({"objects": {"spec": {
        "name": "m", "target_arch": "sm_80",
        "inputs": [
            {"name": "a", "dtype": "fp32", "shape": ["N", "N"], "layout_hint": "any"},
            {"name": "b", "dtype": "fp32", "shape": ["N", "N"], "layout_hint": "any"},
        ],
        "outputs": [{"name": "c", "dtype": "fp32", "shape": ["N", "N"], "layout_hint": "any"}],
        "precision_tolerance": {"rtol": 1e-3, "atol": 1e-3},
        "optimization_priority": "throughput", "notes": "",
    }}}))

    path = _CHECK_PATH.parent / "validate_export.py"
    spec = importlib.util.spec_from_file_location("_validate_export2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module._perf_args(tmp_path)
    assert "--bench-size" in args and args[args.index("--bench-size") + 1] == "4096"
    assert "--expect-ms" in args and args[args.index("--expect-ms") + 1] == "1.5"
    assert args[args.index("--iters") + 1] == "100"
