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
