import pickle
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

from cuda_engine.config import SynthesisConfig
from cuda_engine.services.gpu.local import LocalGPURunner


def test_local_gpu_runner_cache_hit_skips_second_compile(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/cuda/bin/nvcc")

    def fake_run(cmd, capture_output, text, timeout, check):
        calls.append(cmd)
        out_path = Path(cmd[cmd.index("-o") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"so")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    runner = LocalGPURunner(SynthesisConfig(artifact_root=".test_artifacts/gpu"))
    src = f"__global__ void k_{uuid.uuid4().hex}() {{}}"

    first = runner.compile(src, target_arch="sm_80", extra_flags=("--foo",))
    second = runner.compile(src, target_arch="sm_80", extra_flags=("--foo",))

    assert first.ok is True
    assert second.ok is True
    assert runner.cache_hits == 1
    assert len(calls) == 1
    assert "-arch=sm_80" in calls[0]
    assert "--foo" in calls[0]


def test_local_gpu_runner_compile_failure_populates_errors(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/cuda/bin/nvcc")

    def fake_run(cmd, capture_output, text, timeout, check):
        return SimpleNamespace(returncode=1, stdout="", stderr="broken.cu:1: error: bad kernel")

    monkeypatch.setattr("subprocess.run", fake_run)
    runner = LocalGPURunner(SynthesisConfig(artifact_root=".test_artifacts/gpu"))

    result = runner.compile("broken", target_arch="sm_80")

    assert result.ok is False
    assert result.errors == ["broken.cu:1: error: bad kernel"]


def test_local_gpu_runner_raises_clear_error_when_nvcc_missing(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    runner = LocalGPURunner(SynthesisConfig(artifact_root=".test_artifacts/gpu"))

    result = runner.compile("src", target_arch="sm_80")

    assert result.ok is False
    assert "nvcc not found" in result.errors[0]


def test_local_gpu_runner_run_kernel_uses_subprocess_and_reads_outputs(monkeypatch) -> None:
    seen = {}

    def fake_run(cmd, capture_output, text, timeout, check):
        seen["cmd"] = cmd
        seen["timeout"] = timeout
        output_path = Path(cmd[cmd.index("--output") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump({"ok": True, "outputs": ["out"], "stdout": "child out", "stderr": ""}, f)
        return SimpleNamespace(returncode=0, stdout="parent out", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    runner = LocalGPURunner(SynthesisConfig(artifact_root=".test_artifacts/gpu"))

    result = runner.run_kernel(Path("/tmp/kernel.so"), inputs=["in"], timeout_seconds=7)

    assert result.ok is True
    assert result.output_tensors == ["out"]
    assert result.stdout == "child out\nparent out"
    assert seen["timeout"] == 7
    assert "cuda_engine.services.gpu._run_kernel_child" in seen["cmd"]


def test_local_gpu_runner_run_kernel_reports_child_failure(monkeypatch) -> None:
    def fake_run(cmd, capture_output, text, timeout, check):
        output_path = Path(cmd[cmd.index("--output") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump({"ok": False, "outputs": None, "stdout": "", "stderr": "boom"}, f)
        return SimpleNamespace(returncode=0, stdout="", stderr="parent err")

    monkeypatch.setattr("subprocess.run", fake_run)
    runner = LocalGPURunner(SynthesisConfig(artifact_root=".test_artifacts/gpu"))

    result = runner.run_kernel(Path("/tmp/kernel.so"), inputs=[], timeout_seconds=7)

    assert result.ok is False
    assert result.stderr == "boom\nparent err"


def test_local_gpu_runner_run_kernel_timeout(monkeypatch) -> None:
    def fake_run(cmd, capture_output, text, timeout, check):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout, output="out", stderr="err")

    monkeypatch.setattr("subprocess.run", fake_run)
    runner = LocalGPURunner(SynthesisConfig(artifact_root=".test_artifacts/gpu"))

    result = runner.run_kernel(Path("/tmp/kernel.so"), inputs=[], timeout_seconds=2)

    assert result.ok is False
    assert result.timed_out is True
    assert "timed out after 2s" in result.stderr
