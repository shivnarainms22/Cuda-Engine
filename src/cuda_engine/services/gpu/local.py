import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any

from cuda_engine.config import SynthesisConfig
from cuda_engine.services.gpu.base import CompileResult, GPURunner, NsightMetrics, RunResult


class LocalGPURunner(GPURunner):
    def __init__(self, cfg: SynthesisConfig | None = None) -> None:
        self.cfg = cfg or SynthesisConfig()
        root = (
            Path(self.cfg.artifact_root).expanduser() / "compile_cache"
            if self.cfg.artifact_root is not None
            else Path.home() / ".cache" / "cuda_engine" / "compile_cache"
        )
        self.cache_root = root.resolve()
        self.cache_hits = 0

    def compile(
        self,
        src: str,
        *,
        target_arch: str,
        extra_flags: tuple[str, ...] = (),
    ) -> CompileResult:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            return CompileResult(ok=False, errors=["nvcc not found on PATH"], log="nvcc not found")

        flags = (*self.cfg.nvcc_flags, *extra_flags)
        cache_key = self._cache_key(src=src, target_arch=target_arch, flags=flags)
        entry_dir = self.cache_root / cache_key
        so_path = entry_dir / "kernel.so"
        cu_path = entry_dir / "kernel.cu"
        log_path = entry_dir / "compile.log"

        if so_path.exists():
            self.cache_hits += 1
            return CompileResult(ok=True, so_path=so_path, log=log_path.read_text(encoding="utf-8"))

        entry_dir.mkdir(parents=True, exist_ok=True)
        cu_path.write_text(src, encoding="utf-8")
        cmd = [
            nvcc,
            "-shared",
            "-Xcompiler",
            "-fPIC",
            f"-arch={target_arch}",
            *flags,
            "-o",
            str(so_path),
            str(cu_path),
        ]
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.cfg.request_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            log = f"nvcc timed out after {self.cfg.request_timeout_seconds}s\n{exc}"
            log_path.write_text(log, encoding="utf-8")
            return CompileResult(ok=False, log=log, errors=[log])
        except OSError as exc:
            log = str(exc)
            log_path.write_text(log, encoding="utf-8")
            return CompileResult(ok=False, log=log, errors=[log])

        log = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
        log_path.write_text(log, encoding="utf-8")
        if completed.returncode != 0:
            return CompileResult(ok=False, log=log, errors=_extract_error_lines(log))
        return CompileResult(
            ok=True,
            so_path=so_path,
            log=log,
            warnings=_extract_warning_lines(log),
            ptx_size_bytes=so_path.stat().st_size if so_path.exists() else 0,
        )

    def run_kernel(
        self,
        so_path: Path,
        inputs: list[Any],
        timeout_seconds: int = 30,
    ) -> RunResult:
        raise NotImplementedError("LocalGPURunner lands in M1")

    def profile(self, so_path: Path, inputs: list[Any]) -> NsightMetrics:
        raise NotImplementedError("LocalGPURunner lands in M1")

    @staticmethod
    def _cache_key(*, src: str, target_arch: str, flags: tuple[str, ...]) -> str:
        hasher = hashlib.blake2b(digest_size=16)
        hasher.update(target_arch.encode())
        hasher.update(b"\0")
        hasher.update("\0".join(flags).encode())
        hasher.update(b"\0")
        hasher.update(src.encode())
        return hasher.hexdigest()


def _extract_error_lines(log: str) -> list[str]:
    lines = [line.strip() for line in log.splitlines() if "error" in line.lower()]
    return lines or ([log.strip()] if log.strip() else ["nvcc failed"])


def _extract_warning_lines(log: str) -> list[str]:
    return [line.strip() for line in log.splitlines() if "warning" in line.lower()]
