"""Behaviour every ArtifactStore implementation must share."""

from __future__ import annotations

from pathlib import Path

import pytest

from cuda_engine.config import SynthesisConfig
from cuda_engine.services.store.local_dir import LocalDirStore
from cuda_engine.services.store.mocks import InMemoryStore


def test_read_bytes_round_trips_binary_content(tmp_path: Path) -> None:
    """Binary artifacts (a compiled .so) must be readable through the interface."""
    payload = b"\x7fELF\x00\x01\x02\xff"
    for store in (InMemoryStore(), LocalDirStore(SynthesisConfig(artifact_root=str(tmp_path)))):
        run_id = store.new_run()
        store.write_bytes(run_id, "final/kernel.so", payload)
        assert store.read_bytes(run_id, "final/kernel.so") == payload


def test_read_bytes_raises_for_a_missing_file(tmp_path: Path) -> None:
    for store in (InMemoryStore(), LocalDirStore(SynthesisConfig(artifact_root=str(tmp_path)))):
        run_id = store.new_run()
        with pytest.raises(FileNotFoundError):
            store.read_bytes(run_id, "nope.so")
