import pytest

from cuda_engine.services.store.local_dir import LocalDirStore
from cuda_engine.services.store.mocks import InMemoryStore

# ---- InMemoryStore ----

def test_inmemory_exists_false_for_missing_path() -> None:
    store = InMemoryStore()
    run_id = store.new_run()
    assert store.exists(run_id, "checkpoint.json") is False


def test_inmemory_exists_true_after_write_json() -> None:
    store = InMemoryStore()
    run_id = store.new_run()
    store.write_json(run_id, "checkpoint.json", {"a": 1})
    assert store.exists(run_id, "checkpoint.json") is True


def test_inmemory_read_json_returns_written_data() -> None:
    store = InMemoryStore()
    run_id = store.new_run()
    store.write_json(run_id, "checkpoint.json", {"a": 1})
    result = store.read_json(run_id, "checkpoint.json")
    assert result == {"a": 1}


def test_inmemory_read_json_raises_for_missing_path() -> None:
    store = InMemoryStore()
    run_id = store.new_run()
    with pytest.raises(FileNotFoundError):
        store.read_json(run_id, "missing.json")


def test_inmemory_exists_false_for_unknown_run_id() -> None:
    store = InMemoryStore()
    assert store.exists("nonexistent", "file.json") is False


# ---- LocalDirStore ----

def test_local_exists_false_for_missing_path(tmp_path) -> None:
    from cuda_engine.config import SynthesisConfig
    cfg = SynthesisConfig(artifact_root=str(tmp_path))
    store = LocalDirStore(cfg)
    run_id = store.new_run()
    assert store.exists(run_id, "checkpoint.json") is False


def test_local_exists_true_after_write_json(tmp_path) -> None:
    from cuda_engine.config import SynthesisConfig
    cfg = SynthesisConfig(artifact_root=str(tmp_path))
    store = LocalDirStore(cfg)
    run_id = store.new_run()
    store.write_json(run_id, "checkpoint.json", {"a": 1})
    assert store.exists(run_id, "checkpoint.json") is True


def test_local_read_json_returns_written_data(tmp_path) -> None:
    from cuda_engine.config import SynthesisConfig
    cfg = SynthesisConfig(artifact_root=str(tmp_path))
    store = LocalDirStore(cfg)
    run_id = store.new_run()
    store.write_json(run_id, "checkpoint.json", {"a": 1})
    result = store.read_json(run_id, "checkpoint.json")
    assert result == {"a": 1}


def test_local_read_json_raises_for_missing_path(tmp_path) -> None:
    from cuda_engine.config import SynthesisConfig
    cfg = SynthesisConfig(artifact_root=str(tmp_path))
    store = LocalDirStore(cfg)
    run_id = store.new_run()
    with pytest.raises(FileNotFoundError):
        store.read_json(run_id, "missing.json")
