from pathlib import Path

from pydantic import BaseModel, ConfigDict


class KernelArtifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    kernel_cu_path: Path
    kernel_so_path: Path | None = None
    compile_log: str = ""
    ptx_size_bytes: int = 0
