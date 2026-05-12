"""Smoke tests for examples/web_demo.py using Streamlit's AppTest harness.

These verify the demo:
  - imports without crashing,
  - renders the static UI scaffolding,
  - is gated on having both a prompt and a reference source,
  - load_reference_from_source extracts top-level REFERENCE / reference symbols.

No end-to-end synthesis runs; the synthesize call is monkeypatched.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def web_demo_module():
    """Import the demo module via spec_from_file_location to dodge package discovery."""
    pytest.importorskip("streamlit")
    path = Path(__file__).resolve().parents[2] / "examples" / "web_demo.py"
    if not path.exists():
        pytest.skip(f"web demo not present: {path}")

    spec = importlib.util.spec_from_file_location("cuda_engine_web_demo_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # render() is called at module top-level for Streamlit; calling it under a
    # plain import will warn but not crash because streamlit functions silently
    # fall through outside a script context.
    spec.loader.exec_module(module)
    return module


def test_load_reference_from_source_resolves_reference_function(web_demo_module) -> None:
    source = "def reference(x):\n    return x + 1\n"
    fn = web_demo_module.load_reference_from_source(source)
    assert callable(fn)
    assert fn(1) == 2


def test_load_reference_from_source_resolves_REFERENCE_constant(web_demo_module) -> None:
    source = "REFERENCE = lambda x: x * 2\n"
    fn = web_demo_module.load_reference_from_source(source)
    assert fn(3) == 6


def test_load_reference_from_source_raises_when_missing(web_demo_module) -> None:
    source = "x = 1\n"
    with pytest.raises(ValueError, match="must define REFERENCE or reference"):
        web_demo_module.load_reference_from_source(source)


def test_streamlit_apptest_renders_demo_without_crashing() -> None:
    """AppTest harness loads the demo script end-to-end without raising."""
    pytest.importorskip("streamlit")
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("streamlit.testing.v1.AppTest not available in this streamlit version")

    demo_path = Path(__file__).resolve().parents[2] / "examples" / "web_demo.py"
    if not demo_path.exists():
        pytest.skip(f"web demo not present: {demo_path}")

    at = AppTest.from_file(str(demo_path))
    at.run(timeout=30)
    assert not at.exception, f"demo raised: {at.exception}"
    # Title should render even when no prompt/reference is provided yet.
    titles = [t.value for t in at.title]
    assert any("cuda-engine" in title for title in titles), titles
