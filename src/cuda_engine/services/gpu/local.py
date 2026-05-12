import csv
import hashlib
import io
import pickle
import shutil
import subprocess
import sys
import sysconfig
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from cuda_engine.config import SynthesisConfig
from cuda_engine.services.gpu.base import (
    BenchmarkResult,
    CompileResult,
    GPURunner,
    NsightMetrics,
    RunResult,
)


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

        flags = (*self.cfg.nvcc_flags, *extra_flags, *_torch_extension_flags(src))
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
        run_dir = self.cache_root / "run_tmp" / uuid.uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=True)
        input_path = run_dir / "inputs.pkl"
        output_path = run_dir / "outputs.pkl"
        with input_path.open("wb") as f:
            pickle.dump(inputs, f)

        cmd = [
            sys.executable,
            "-m",
            "cuda_engine.services.gpu._run_kernel_child",
            "--so",
            str(so_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
        started_at = time.time()
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _stringify_timeout_stream(exc.output)
            stderr = _stringify_timeout_stream(exc.stderr)
            return RunResult(
                ok=False,
                stdout=stdout,
                stderr=f"run_kernel timed out after {timeout_seconds}s\n{stderr}".strip(),
                timed_out=True,
                wall_seconds=time.time() - started_at,
            )

        if not output_path.exists():
            return RunResult(
                ok=False,
                stdout=completed.stdout,
                stderr=completed.stderr or "run_kernel child produced no output payload",
                wall_seconds=time.time() - started_at,
            )

        try:
            with output_path.open("rb") as f:
                payload = pickle.load(f)
        except (EOFError, pickle.PickleError, OSError) as exc:
            return RunResult(
                ok=False,
                stdout=completed.stdout,
                stderr=_join_streams(
                    f"run_kernel child could not decode output payload: {exc}",
                    completed.stderr,
                ),
                wall_seconds=time.time() - started_at,
            )
        child_stdout = str(payload.get("stdout", ""))
        child_stderr = str(payload.get("stderr", ""))
        return RunResult(
            ok=bool(payload.get("ok", False)) and completed.returncode == 0,
            output_tensors=payload.get("outputs"),
            stdout=_join_streams(child_stdout, completed.stdout),
            stderr=_join_streams(child_stderr, completed.stderr),
            timed_out=False,
            wall_seconds=time.time() - started_at,
        )

    def benchmark_kernel(
        self,
        so_path: Path,
        inputs: list[Any],
        *,
        reference: Callable[..., Any] | None = None,
        warmup_iterations: int = 10,
        timed_iterations: int = 50,
        timeout_seconds: int = 60,
    ) -> BenchmarkResult:
        run_dir = self.cache_root / "run_tmp" / uuid.uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=True)
        input_path = run_dir / "inputs.pkl"
        output_path = run_dir / "benchmark.pkl"
        with input_path.open("wb") as f:
            pickle.dump({"inputs": inputs, "reference": reference}, f)

        cmd = [
            sys.executable,
            "-m",
            "cuda_engine.services.gpu._run_kernel_child",
            "--so",
            str(so_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--benchmark",
            "--warmup-iterations",
            str(warmup_iterations),
            "--timed-iterations",
            str(timed_iterations),
        ]
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _stringify_timeout_stream(exc.output)
            stderr = _stringify_timeout_stream(exc.stderr)
            return BenchmarkResult(
                ok=False,
                stdout=stdout,
                stderr=f"benchmark_kernel timed out after {timeout_seconds}s\n{stderr}".strip(),
                timed_out=True,
                warmup_iterations=warmup_iterations,
                timed_iterations=timed_iterations,
            )

        if not output_path.exists():
            return BenchmarkResult(
                ok=False,
                stdout=completed.stdout,
                stderr=completed.stderr or "benchmark child produced no output payload",
                warmup_iterations=warmup_iterations,
                timed_iterations=timed_iterations,
            )

        with output_path.open("rb") as f:
            payload = pickle.load(f)
        benchmark_payload = payload.get("benchmark")
        if not isinstance(benchmark_payload, dict):
            return BenchmarkResult(
                ok=False,
                stdout=_join_streams(str(payload.get("stdout", "")), completed.stdout),
                stderr=_join_streams(str(payload.get("stderr", "")), completed.stderr),
                warmup_iterations=warmup_iterations,
                timed_iterations=timed_iterations,
            )
        benchmark = BenchmarkResult.model_validate(benchmark_payload)
        return benchmark.model_copy(
            update={
                "ok": benchmark.ok and completed.returncode == 0,
                "stdout": _join_streams(str(payload.get("stdout", "")), completed.stdout),
                "stderr": _join_streams(str(payload.get("stderr", "")), completed.stderr),
            }
        )

    def profile(self, so_path: Path, inputs: list[Any]) -> NsightMetrics:
        ncu = shutil.which("ncu")
        if ncu is None:
            return NsightMetrics(raw_csv="ncu_not_available")

        run_dir = self.cache_root / "profile_tmp" / uuid.uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=True)
        input_path = run_dir / "inputs.pkl"
        output_path = run_dir / "outputs.pkl"
        with input_path.open("wb") as f:
            pickle.dump(inputs, f)

        cmd = [
            ncu,
            "--csv",
            "--set",
            "basic",
            "--target-processes",
            "all",
            "--",
            sys.executable,
            "-m",
            "cuda_engine.services.gpu._run_kernel_child",
            "--so",
            str(so_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
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
            return NsightMetrics(raw_csv=f"ncu_timeout: {exc}")

        if completed.returncode != 0:
            return NsightMetrics(raw_csv=completed.stderr or completed.stdout or "ncu_failed")
        metrics = parse_ncu_csv(completed.stdout)
        if metrics.occupancy is None and metrics.regs_per_thread is None:
            child_stderr = _read_child_stderr(output_path)
            if child_stderr:
                metrics = metrics.model_copy(
                    update={"raw_csv": f"{child_stderr}\n---\n{metrics.raw_csv}"}
                )
        return metrics

    @staticmethod
    def _cache_key(*, src: str, target_arch: str, flags: tuple[str, ...]) -> str:
        hasher = hashlib.blake2b(digest_size=16)
        hasher.update(target_arch.encode())
        hasher.update(b"\0")
        hasher.update("\0".join(flags).encode())
        hasher.update(b"\0")
        hasher.update(src.encode())
        return hasher.hexdigest()


def _torch_extension_flags(src: str = "") -> tuple[str, ...]:
    if "torch/extension.h" not in src and "TORCH_LIBRARY" not in src:
        return ()
    try:
        import torch
        from torch.utils.cpp_extension import include_paths, library_paths
    except ImportError:
        return ()

    flags: list[str] = ["-std=c++17", "-DTORCH_API_INCLUDE_EXTENSION_H"]
    flags.extend(f"-I{path}" for path in include_paths(device_type="cuda"))
    python_include = sysconfig.get_paths().get("include")
    if python_include:
        flags.append(f"-I{python_include}")
    flags.extend(f"-L{path}" for path in library_paths(device_type="cuda"))
    flags.extend(("-lc10", "-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10_cuda"))
    abi = int(cast(Any, torch)._C._GLIBCXX_USE_CXX11_ABI)
    flags.append(f"-D_GLIBCXX_USE_CXX11_ABI={abi}")
    return tuple(flags)


def _extract_error_lines(log: str) -> list[str]:
    lines = [line.strip() for line in log.splitlines() if "error" in line.lower()]
    return lines or ([log.strip()] if log.strip() else ["nvcc failed"])


def _extract_warning_lines(log: str) -> list[str]:
    return [line.strip() for line in log.splitlines() if "warning" in line.lower()]


def _join_streams(*parts: str) -> str:
    return "\n".join(part for part in (p.strip() for p in parts) if part)


def _stringify_timeout_stream(stream: str | bytes | None) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode(errors="replace")
    return stream


def parse_ncu_csv(csv_text: str) -> NsightMetrics:
    header_index = _find_csv_header(csv_text)
    if header_index is None:
        return NsightMetrics(raw_csv=csv_text)

    reader = csv.DictReader(io.StringIO(csv_text[header_index:]))
    occupancy: float | None = None
    regs_per_thread: int | None = None
    achieved_bandwidth_gbps: float | None = None
    first_id: str | None = None
    for row in reader:
        row_id = row.get("ID", "")
        if first_id is None:
            first_id = row_id
        if row_id != first_id:
            continue
        section = (row.get("Section Name") or "").strip()
        metric = (row.get("Metric Name") or "").strip()
        unit = (row.get("Metric Unit") or "").strip()
        value = (row.get("Metric Value") or "").strip()
        if not value:
            continue
        if section == "Occupancy" and metric == "Achieved Occupancy" and occupancy is None:
            occupancy = _parse_float(value) / 100.0
        elif (
            section == "Launch Statistics"
            and metric == "Registers Per Thread"
            and regs_per_thread is None
        ):
            regs_per_thread = int(_parse_float(value))
        elif achieved_bandwidth_gbps is None:
            achieved_bandwidth_gbps = _parse_bandwidth_gbps(
                section=section,
                metric=metric,
                unit=unit,
                value=value,
            )

    return NsightMetrics(
        occupancy=occupancy,
        regs_per_thread=regs_per_thread,
        achieved_bandwidth_gbps=achieved_bandwidth_gbps,
        raw_csv=csv_text,
    )


def _find_csv_header(csv_text: str) -> int | None:
    marker = '"ID","Process ID"'
    index = csv_text.find(marker)
    return index if index >= 0 else None


def _parse_float(value: str) -> float:
    return float(value.replace(",", ""))


def _parse_bandwidth_gbps(*, section: str, metric: str, unit: str, value: str) -> float | None:
    section_lower = section.lower()
    metric_lower = metric.lower()
    unit_lower = unit.lower()
    if "memory" not in section_lower and "dram" not in section_lower and "l2" not in section_lower:
        return None
    if "throughput" not in metric_lower and "bandwidth" not in metric_lower:
        return None

    bandwidth = _parse_float(value)
    if unit_lower in {"gbyte/second", "gb/s", "gbyte/s"}:
        return bandwidth
    if unit_lower in {"mbyte/second", "mb/s", "mbyte/s"}:
        return bandwidth / 1000.0
    return None


def _read_child_stderr(output_path: Path) -> str:
    if not output_path.exists():
        return ""
    try:
        with output_path.open("rb") as f:
            payload = pickle.load(f)
    except (EOFError, pickle.PickleError, OSError):
        return ""
    if payload.get("ok") is True:
        return ""
    return str(payload.get("stderr") or "").strip()
