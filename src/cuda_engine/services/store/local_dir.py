import json
import uuid
from pathlib import Path

from pydantic import BaseModel

from cuda_engine.config import SynthesisConfig
from cuda_engine.services.store.base import ArtifactStore


class LocalDirStore(ArtifactStore):
    def __init__(self, cfg: SynthesisConfig | None = None) -> None:
        self.cfg = cfg or SynthesisConfig()
        self.root = (
            Path(self.cfg.artifact_root).expanduser()
            if self.cfg.artifact_root is not None
            else Path.home() / ".cache" / "cuda_engine" / "runs"
        ).resolve()

    def new_run(self) -> str:
        run_id = uuid.uuid4().hex[:12]
        self.run_dir(run_id).mkdir(parents=True, exist_ok=False)
        return run_id

    def run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def write_text(self, run_id: str, rel_path: str, content: str) -> Path:
        path = self.run_dir(run_id) / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def write_bytes(self, run_id: str, rel_path: str, content: bytes) -> Path:
        path = self.run_dir(run_id) / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def write_json(self, run_id: str, rel_path: str, obj: object) -> Path:
        payload: object = obj.model_dump(mode="json") if isinstance(obj, BaseModel) else obj
        return self.write_text(run_id, rel_path, json.dumps(payload, default=str, indent=2))
