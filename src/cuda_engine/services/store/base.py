from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


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

    @abstractmethod
    def exists(self, run_id: str, rel_path: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def read_json(self, run_id: str, rel_path: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def read_text(self, run_id: str, rel_path: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def read_bytes(self, run_id: str, rel_path: str) -> bytes:
        """Read a binary artifact (e.g. a compiled .so) written by this store."""
        raise NotImplementedError

    def rel_path_of(self, run_id: str, path: Path) -> str | None:
        """The rel_path of an artifact path within this run, or None if the path
        is not under this store's run dir (i.e. a plain filesystem path a caller
        should read directly). Lets callers read a written artifact back through
        the store without knowing its path format."""
        try:
            return path.relative_to(self.run_dir(run_id)).as_posix()
        except ValueError:
            return None
