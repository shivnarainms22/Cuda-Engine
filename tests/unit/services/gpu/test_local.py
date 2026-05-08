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


def test_local_gpu_runner_adds_torch_extension_flags_for_custom_op(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/cuda/bin/nvcc")
    monkeypatch.setattr(
        "cuda_engine.services.gpu.local._torch_extension_flags",
        lambda src="": (
            "-I/torch/include",
            "-I/python/include",
            "-L/torch/lib",
            "-ltorch",
        ),
    )

    def fake_run(cmd, capture_output, text, timeout, check):
        calls.append(cmd)
        out_path = Path(cmd[cmd.index("-o") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"so")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    runner = LocalGPURunner(SynthesisConfig(artifact_root=".test_artifacts/gpu"))
    src = f'#include <torch/extension.h>\nTORCH_LIBRARY(x_{uuid.uuid4().hex}, m) {{}}'

    result = runner.compile(src, target_arch="sm_80")

    assert result.ok is True
    assert "-I/torch/include" in calls[0]
    assert "-I/python/include" in calls[0]
    assert "-L/torch/lib" in calls[0]
    assert "-ltorch" in calls[0]


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


def test_local_gpu_runner_run_kernel_reports_corrupt_child_payload(monkeypatch) -> None:
    def fake_run(cmd, capture_output, text, timeout, check):
        output_path = Path(cmd[cmd.index("--output") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"")
        return SimpleNamespace(returncode=1, stdout="parent out", stderr="child crashed")

    monkeypatch.setattr("subprocess.run", fake_run)
    runner = LocalGPURunner(SynthesisConfig(artifact_root=".test_artifacts/gpu"))

    result = runner.run_kernel(Path("/tmp/kernel.so"), inputs=[], timeout_seconds=7)

    assert result.ok is False
    assert "could not decode output payload" in result.stderr
    assert "child crashed" in result.stderr


def test_local_gpu_runner_run_kernel_timeout(monkeypatch) -> None:
    def fake_run(cmd, capture_output, text, timeout, check):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout, output="out", stderr="err")

    monkeypatch.setattr("subprocess.run", fake_run)
    runner = LocalGPURunner(SynthesisConfig(artifact_root=".test_artifacts/gpu"))

    result = runner.run_kernel(Path("/tmp/kernel.so"), inputs=[], timeout_seconds=2)

    assert result.ok is False
    assert result.timed_out is True
    assert "timed out after 2s" in result.stderr


def test_local_gpu_runner_benchmark_kernel_uses_subprocess_and_reads_payload(monkeypatch) -> None:
    seen = {}

    def fake_run(cmd, capture_output, text, timeout, check):
        seen["cmd"] = cmd
        output_path = Path(cmd[cmd.index("--output") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump(
                {
                    "ok": True,
                    "benchmark": {
                        "ok": True,
                        "custom_ms": 0.25,
                        "baseline_ms": 1.0,
                        "warmup_iterations": 3,
                        "timed_iterations": 11,
                    },
                    "stdout": "",
                    "stderr": "",
                },
                f,
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    runner = LocalGPURunner(SynthesisConfig(artifact_root=".test_artifacts/gpu"))

    result = runner.benchmark_kernel(
        Path("/tmp/kernel.so"),
        inputs=["in"],
        warmup_iterations=3,
        timed_iterations=11,
        timeout_seconds=9,
    )

    assert result.ok is True
    assert result.custom_ms == 0.25
    assert result.baseline_ms == 1.0
    assert "--benchmark" in seen["cmd"]
    assert "11" in seen["cmd"]


def test_local_gpu_runner_benchmark_kernel_reports_child_failure(monkeypatch) -> None:
    def fake_run(cmd, capture_output, text, timeout, check):
        output_path = Path(cmd[cmd.index("--output") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump({"ok": False, "stdout": "", "stderr": "benchmark boom"}, f)
        return SimpleNamespace(returncode=0, stdout="", stderr="parent err")

    monkeypatch.setattr("subprocess.run", fake_run)
    runner = LocalGPURunner(SynthesisConfig(artifact_root=".test_artifacts/gpu"))

    result = runner.benchmark_kernel(Path("/tmp/kernel.so"), inputs=[], timeout_seconds=9)

    assert result.ok is False
    assert result.stderr == "benchmark boom\nparent err"


def test_profile_returns_unavailable_metrics_when_ncu_missing(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    runner = LocalGPURunner(SynthesisConfig(artifact_root=".test_artifacts/gpu"))

    metrics = runner.profile(Path("/tmp/kernel.so"), inputs=[])

    assert metrics.occupancy is None
    assert metrics.regs_per_thread is None
    assert metrics.uncoalesced_global_loads_pct is None
    assert metrics.spill_bytes == 0
    assert metrics.achieved_bandwidth_gbps is None
    assert metrics.achieved_tflops is None
    assert metrics.raw_csv == "ncu_not_available"


def test_profile_invokes_ncu_and_parses_stdout(monkeypatch, tmp_path) -> None:
    fixture_csv = (
        '"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream",'
        '"Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit",'
        '"Metric Value","Rule Name","Rule Type","Rule Description",'
        '"Estimated Speedup Type","Estimated Speedup"\n'
        '"0","1","p","h","k","1","7","(256,1,1)","(4,1,1)","0","8.0","Occupancy",'
        '"Achieved Occupancy","%","80.00","","","","",""\n'
        '"0","1","p","h","k","1","7","(256,1,1)","(4,1,1)","0","8.0","Launch Statistics",'
        '"Registers Per Thread","register/thread","24","","","","",""\n'
    )
    captured_cmd: list[list[str]] = []

    def fake_which(name):
        return "/usr/local/cuda/bin/ncu" if name == "ncu" else None

    def fake_run(cmd, capture_output, text, timeout, check):
        captured_cmd.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout=fixture_csv, stderr="")

    monkeypatch.setattr("shutil.which", fake_which)
    monkeypatch.setattr("subprocess.run", fake_run)

    so_path = tmp_path / "kernel.so"
    so_path.write_bytes(b"")
    runner = LocalGPURunner(SynthesisConfig(artifact_root=str(tmp_path)))

    metrics = runner.profile(so_path, inputs=[1, 2, 3])

    assert abs(metrics.occupancy - 0.80) < 1e-6
    assert metrics.regs_per_thread == 24
    assert metrics.raw_csv == fixture_csv
    assert captured_cmd, "expected ncu subprocess to be invoked"
    cmd = captured_cmd[0]
    assert cmd[0] == "/usr/local/cuda/bin/ncu"
    assert "--csv" in cmd
    assert "--set" in cmd and "basic" in cmd
    assert "--target-processes" in cmd and "all" in cmd
    assert str(so_path) in cmd


def test_profile_returns_unavailable_metrics_when_ncu_subprocess_fails(
    monkeypatch, tmp_path
) -> None:
    def fake_which(name):
        return "/usr/local/cuda/bin/ncu" if name == "ncu" else None

    def fake_run(cmd, capture_output, text, timeout, check):
        return SimpleNamespace(returncode=2, stdout="", stderr="ERR_NVGPUCTRPERM")

    monkeypatch.setattr("shutil.which", fake_which)
    monkeypatch.setattr("subprocess.run", fake_run)

    so_path = tmp_path / "kernel.so"
    so_path.write_bytes(b"")
    runner = LocalGPURunner(SynthesisConfig(artifact_root=str(tmp_path)))

    metrics = runner.profile(so_path, inputs=[])

    assert metrics.occupancy is None
    assert metrics.regs_per_thread is None
    assert "ERR_NVGPUCTRPERM" in metrics.raw_csv


def test_profile_surfaces_child_traceback_when_ncu_reports_no_kernels(
    monkeypatch, tmp_path
) -> None:
    ncu_stdout = (
        "==PROF== Connected to process 1\n"
        "==PROF== Disconnected from process 1\n"
        "==WARNING== No kernels were profiled.\n"
    )
    child_traceback = (
        "Traceback (most recent call last):\n"
        '  File "_run_kernel_child.py", line 84, in _torch_custom_op_forward\n'
        "    return cast(Any, torch.ops.cuda_engine.forward)\n"
        "AttributeError: '_OpNamespace' 'cuda_engine' object has no attribute 'forward'\n"
    )

    def fake_which(name):
        return "/usr/local/cuda/bin/ncu" if name == "ncu" else None

    def fake_run(cmd, capture_output, text, timeout, check):
        # The child wrote its failure pickle to the path passed via --output.
        output_index = cmd.index("--output") + 1
        output_path = Path(cmd[output_index])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump(
                {"ok": False, "outputs": None, "stdout": "", "stderr": child_traceback},
                f,
            )
        return SimpleNamespace(returncode=0, stdout=ncu_stdout, stderr="")

    monkeypatch.setattr("shutil.which", fake_which)
    monkeypatch.setattr("subprocess.run", fake_run)

    so_path = tmp_path / "kernel.so"
    so_path.write_bytes(b"")
    runner = LocalGPURunner(SynthesisConfig(artifact_root=str(tmp_path)))

    metrics = runner.profile(so_path, inputs=[])

    assert metrics.occupancy is None
    assert "AttributeError" in metrics.raw_csv
    assert "No kernels were profiled" in metrics.raw_csv
