from abc import ABC, abstractmethod
from pathlib import Path


class ArtifactStore(ABC):
    @abstractmethod
    def new_run(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def run_dir(self, run_id: str) -> Path:
        raise NotImplementedError

    @abstractmethod
    def write_text(self, run_id: str, rel_path: str, content: str) -> Path:
        raise NotImplementedError

    @abstractmethod
    def write_bytes(self, run_id: str, rel_path: str, content: bytes) -> Path:
        raise NotImplementedError

    @abstractmethod
    def write_json(self, run_id: str, rel_path: str, obj: object) -> Path:
        raise NotImplementedError
