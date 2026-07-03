import hashlib
from typing import Any

from pydantic import BaseModel, Field


class Checkpoint(BaseModel):
    inputs_fingerprint: str
    completed_stages: list[str] = Field(default_factory=list)
    objects: dict[str, Any] = Field(default_factory=dict)


def compute_fingerprint(*, prompt: str, reference_src: str, target: str) -> str:
    """SHA-256 hex digest over prompt, reference_src, and target separated by NUL bytes."""
    h = hashlib.sha256()
    h.update(prompt.encode())
    h.update(b"\x00")
    h.update(reference_src.encode())
    h.update(b"\x00")
    h.update(target.encode())
    return h.hexdigest()
