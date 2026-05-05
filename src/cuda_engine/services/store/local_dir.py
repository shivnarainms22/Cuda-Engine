from pathlib import Path

from cuda_engine.config import SynthesisConfig
from cuda_engine.services.store.base import ArtifactStore


class LocalDirStore(ArtifactStore):
    def __init__(self, cfg: SynthesisConfig | None = None) -> None:
        self.cfg = cfg

    def new_run(self) -> str:
        raise NotImplementedError("LocalDirStore lands in M1")

    def run_dir(self, run_id: str) -> Path:
        raise NotImplementedError("LocalDirStore lands in M1")

    def write_text(self, run_id: str, rel_path: str, content: str) -> Path:
        raise NotImplementedError("LocalDirStore lands in M1")

    def write_bytes(self, run_id: str, rel_path: str, content: bytes) -> Path:
        raise NotImplementedError("LocalDirStore lands in M1")

    def write_json(self, run_id: str, rel_path: str, obj: object) -> Path:
        raise NotImplementedError("LocalDirStore lands in M1")
