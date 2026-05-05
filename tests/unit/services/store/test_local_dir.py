import json
from pathlib import Path

from cuda_engine.config import SynthesisConfig
from cuda_engine.models import SynthesisReport
from cuda_engine.services.store.local_dir import LocalDirStore


def test_local_dir_store_creates_short_hex_run_id() -> None:
    store = LocalDirStore(SynthesisConfig(artifact_root=".test_artifacts/store"))

    run_id = store.new_run()

    assert len(run_id) == 12
    int(run_id, 16)
    assert store.run_dir(run_id).is_absolute()
    assert store.run_dir(run_id).exists()


def test_local_dir_store_writes_text_bytes_and_json() -> None:
    store = LocalDirStore(SynthesisConfig(artifact_root=".test_artifacts/store"))
    run_id = store.new_run()

    text_path = store.write_text(run_id, "stage1/prompt.md", "hello")
    bytes_path = store.write_bytes(run_id, "stage1/raw.bin", b"\x00\x01")
    json_path = store.write_json(
        run_id,
        "report.json",
        SynthesisReport(run_id=run_id, spec_name="x", stages_executed=["stage1"]),
    )

    assert text_path.read_text(encoding="utf-8") == "hello"
    assert bytes_path.read_bytes() == b"\x00\x01"
    assert json.loads(json_path.read_text(encoding="utf-8"))["run_id"] == run_id


def test_local_dir_store_default_root_is_cache_dir() -> None:
    store = LocalDirStore(SynthesisConfig())

    assert store.root == Path.home() / ".cache" / "cuda_engine" / "runs"
