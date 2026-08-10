"""The package version must not drift from the distribution metadata."""

from __future__ import annotations

from importlib.metadata import version

import cuda_engine


def test_version_matches_the_installed_distribution() -> None:
    """It was hardcoded to 0.0.1 while PyPI served 1.2.0 -- derive it instead."""
    assert cuda_engine.__version__ == version("cuda-engine")


def test_version_is_not_the_stale_placeholder() -> None:
    assert cuda_engine.__version__ != "0.0.1"
