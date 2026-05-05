import json
import uuid
from pathlib import Path

from cuda_engine.services.store.base import ArtifactStore


class InMemoryStore(ArtifactStore):
    def __init__(self) -> None:
        self._files: dict[tuple[str, str], bytes] = {}

    def new_run(self) -> str:
        return uuid.uuid4().hex[:12]

    def run_dir(self, run_id: str) -> Path:
        return Path(f"<memory>/{run_id}")

    def write_text(self, run_id: str, rel_path: str, content: str) -> Path:
        self._files[(run_id, rel_path)] = content.encode()
        return self.run_dir(run_id) / rel_path

    def write_bytes(self, run_id: str, rel_path: str, content: bytes) -> Path:
        self._files[(run_id, rel_path)] = content
        return self.run_dir(run_id) / rel_path

    def write_json(self, run_id: str, rel_path: str, obj: object) -> Path:
        return self.write_text(run_id, rel_path, json.dumps(obj, default=str, indent=2))
