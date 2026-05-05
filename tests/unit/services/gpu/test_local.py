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
