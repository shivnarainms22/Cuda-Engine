"""Exporting a verified run as a self-contained, importable kernel package.

Runs on CPU-only torch with no nvcc and no LLM call.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

from cuda_engine.export import ExportError, build_export, write_export
from cuda_engine.models import (
    CorrectnessReport,
    KernelSpec,
    OptimizationPriority,
    PerformanceReport,
    PrecisionTolerance,
    SynthesisReport,
    SynthesisResult,
    TensorArg,
)
from cuda_engine.services.store.mocks import InMemoryStore

KERNEL_SRC = "// generated kernel\n#include <torch/extension.h>\n"


def _spec() -> KernelSpec:
    return KernelSpec(
        name="rms_norm_fp16",
        target_arch="sm_80",
        inputs=[TensorArg(name="x", dtype="fp16", shape=("B", "D"))],
        outputs=[TensorArg(name="y", dtype="fp16", shape=("B", "D"))],
        precision_tolerance=PrecisionTolerance(rtol=1e-3, atol=1e-3),
        optimization_priority=OptimizationPriority.BALANCED,
    )


def _populate(
    store: InMemoryStore,
    *,
    passed: bool = True,
    correctness_passed: bool = True,
    polished: bool = True,
    with_so: bool = False,
) -> str:
    run_id = "run123"
    spec = _spec()
    correctness = CorrectnessReport(
        passed=correctness_passed,
        max_abs_err=4.8e-4,
        max_rel_err=9.1e-4,
        shapes_tested=[(1,), (127,), (1024,)],
    )
    performance = PerformanceReport(
        speedup_vs_reference=1.42,
        speedup_vs_torch_compile=1.08,
        achieved_gbps=1290.0,
        below_target=False,
        notes=["baseline_mode=default"],
    )
    result = SynthesisResult(
        passed=passed,
        run_id=run_id,
        artifacts_dir=f"<memory>/{run_id}",
        report=SynthesisReport(run_id=run_id, spec_name=spec.name, stages_executed=["interview"]),
        correctness=correctness,
        performance=performance,
    )
    store.write_json(run_id, "report.json", result.model_dump(mode="json", exclude={"kernel_callable"}))
    store.write_json(
        run_id,
        "checkpoint.json",
        {
            "inputs_fingerprint": "abc",
            "completed_stages": ["polish"],
            "objects": {
                "spec": spec.model_dump(mode="json"),
                "correctness": correctness.model_dump(mode="json"),
                "performance": performance.model_dump(mode="json"),
            },
        },
    )
    store.write_text(run_id, "inputs/prompt.txt", "make an rms norm kernel")
    if polished:
        store.write_text(run_id, "stage5_polish/final/kernel.cu", KERNEL_SRC)
    else:
        store.write_text(run_id, "stage2_codegen/final/kernel.cu", KERNEL_SRC)
    if with_so:
        store.write_bytes(run_id, "stage5_polish/final/kernel.so", b"\x7fELF-not-real")
    return run_id


# --- D2: refuse to export an unverified kernel ------------------------------


def test_refuses_when_correctness_failed() -> None:
    store = InMemoryStore()
    run_id = _populate(store, passed=False, correctness_passed=False)
    with pytest.raises(ExportError, match="correctness"):
        build_export(store, run_id)


def test_refuses_when_report_is_missing() -> None:
    store = InMemoryStore()
    with pytest.raises(ExportError):
        build_export(store, "nonexistent")


def test_refuses_when_no_kernel_source_present() -> None:
    store = InMemoryStore()
    run_id = _populate(store)
    store._files.pop((run_id, "stage5_polish/final/kernel.cu"))
    with pytest.raises(ExportError, match="kernel source"):
        build_export(store, run_id)


def test_force_exports_but_stamps_an_unverified_banner() -> None:
    store = InMemoryStore()
    run_id = _populate(store, passed=False, correctness_passed=False)
    files = build_export(store, run_id, force=True)
    assert "UNVERIFIED" in files["VERIFICATION.md"]


def test_verified_export_has_no_unverified_banner() -> None:
    store = InMemoryStore()
    run_id = _populate(store)
    files = build_export(store, run_id)
    assert "UNVERIFIED" not in files["VERIFICATION.md"]


# --- layout -----------------------------------------------------------------


def test_expected_files_are_present() -> None:
    store = InMemoryStore()
    run_id = _populate(store)
    files = build_export(store, run_id)
    pkg = "ce_rms_norm_fp16"
    assert set(files) == {
        "pyproject.toml",
        "README.md",
        "VERIFICATION.md",
        f"{pkg}/__init__.py",
        f"{pkg}/fake_impl.py",
        f"{pkg}/kernel.cu",
        f"{pkg}/spec.json",
        f"{pkg}/manifest.json",
    }


def test_package_name_is_sanitised_and_prefixed() -> None:
    store = InMemoryStore()
    run_id = "r"
    spec = _spec().model_copy(update={"name": "top-k v2.fp32"})
    store.write_json(
        run_id,
        "report.json",
        SynthesisResult(
            passed=True,
            run_id=run_id,
            artifacts_dir="x",
            report=SynthesisReport(run_id=run_id, spec_name=spec.name, stages_executed=[]),
            correctness=CorrectnessReport(passed=True, max_abs_err=0, max_rel_err=0, shapes_tested=[]),
            performance=PerformanceReport(),
        ).model_dump(mode="json", exclude={"kernel_callable"}),
    )
    store.write_json(run_id, "checkpoint.json", {"inputs_fingerprint": "a", "objects": {"spec": spec.model_dump(mode="json")}})
    store.write_text(run_id, "stage5_polish/final/kernel.cu", KERNEL_SRC)
    files = build_export(store, run_id)
    assert "ce_top_k_v2_fp32/__init__.py" in files


def test_kernel_source_is_copied_verbatim() -> None:
    store = InMemoryStore()
    run_id = _populate(store)
    files = build_export(store, run_id)
    assert files["ce_rms_norm_fp16/kernel.cu"] == KERNEL_SRC


# --- D4: which kernel shipped is recorded -----------------------------------


def test_polished_kernel_is_preferred_and_recorded() -> None:
    import json

    store = InMemoryStore()
    run_id = _populate(store, polished=True)
    files = build_export(store, run_id)
    manifest = json.loads(files["ce_rms_norm_fp16/manifest.json"])
    assert manifest["source_kernel"] == "stage5_polish/final/kernel.cu"
    assert manifest["run_id"] == run_id


def test_falls_back_to_codegen_kernel_and_records_it() -> None:
    import json

    store = InMemoryStore()
    run_id = _populate(store, polished=False)
    files = build_export(store, run_id)
    manifest = json.loads(files["ce_rms_norm_fp16/manifest.json"])
    assert manifest["source_kernel"] == "stage2_codegen/final/kernel.cu"


def test_manifest_records_whether_a_prebuilt_so_is_included() -> None:
    import json

    store = InMemoryStore()
    files = build_export(store, _populate(store, with_so=True))
    assert json.loads(files["ce_rms_norm_fp16/manifest.json"])["prebuilt_so"] is True
    store2 = InMemoryStore()
    files2 = build_export(store2, _populate(store2, with_so=False))
    assert json.loads(files2["ce_rms_norm_fp16/manifest.json"])["prebuilt_so"] is False


# --- generated files are valid ----------------------------------------------


def test_generated_python_is_syntactically_valid() -> None:
    store = InMemoryStore()
    run_id = _populate(store)
    files = build_export(store, run_id)
    compile(files["ce_rms_norm_fp16/__init__.py"], "__init__.py", "exec")
    compile(files["ce_rms_norm_fp16/fake_impl.py"], "fake_impl.py", "exec")


def test_generated_fake_impl_resolves_the_spec_shapes() -> None:
    store = InMemoryStore()
    run_id = _populate(store)
    files = build_export(store, run_id)
    namespace: dict[str, Any] = {}
    exec(files["ce_rms_norm_fp16/fake_impl.py"], namespace)
    assert namespace["resolve_output_shapes"]([(8, 512)]) == [(8, 512)]


def test_pyproject_parses_and_names_the_package() -> None:
    store = InMemoryStore()
    run_id = _populate(store)
    data = tomllib.loads(build_export(store, run_id)["pyproject.toml"])
    assert data["project"]["name"] == "ce-rms-norm-fp16"
    assert "torch" in " ".join(data["project"]["dependencies"])


def test_spec_json_round_trips_to_the_same_spec() -> None:
    store = InMemoryStore()
    run_id = _populate(store)
    files = build_export(store, run_id)
    assert KernelSpec.model_validate_json(files["ce_rms_norm_fp16/spec.json"]) == _spec()


# --- D5: VERIFICATION.md states the negative space --------------------------


def test_verification_records_evidence_and_limits() -> None:
    store = InMemoryStore()
    run_id = _populate(store)
    text = build_export(store, run_id)["VERIFICATION.md"]
    assert "1.08" in text  # speedup vs torch.compile
    assert "sm_80" in text  # target it was verified on
    assert "1024" in text  # a shape actually tested
    assert "0.001" in text or "1e-03" in text  # tolerance
    assert "Not verified" in text  # the negative space section


def test_verification_notes_when_performance_is_below_target() -> None:
    store = InMemoryStore()
    run_id = _populate(store)
    report = store.read_json(run_id, "report.json")
    report["performance"]["below_target"] = True
    report["performance"]["speedup_vs_torch_compile"] = 0.66
    store.write_json(run_id, "report.json", report)
    text = build_export(store, run_id)["VERIFICATION.md"]
    assert "below" in text.lower()
    assert "0.66" in text


# --- write_export -----------------------------------------------------------


def test_write_export_writes_every_file_to_disk(tmp_path: Path) -> None:
    store = InMemoryStore()
    run_id = _populate(store)
    out = write_export(store, run_id, tmp_path / "pkg")
    assert (out / "pyproject.toml").is_file()
    assert (out / "ce_rms_norm_fp16" / "kernel.cu").read_text() == KERNEL_SRC


def test_write_export_copies_the_prebuilt_so(tmp_path: Path) -> None:
    store = InMemoryStore()
    run_id = _populate(store, with_so=True)
    out = write_export(store, run_id, tmp_path / "pkg")
    assert (out / "ce_rms_norm_fp16" / "kernel.so").read_bytes() == b"\x7fELF-not-real"


def test_write_export_refuses_a_non_empty_destination(tmp_path: Path) -> None:
    store = InMemoryStore()
    run_id = _populate(store)
    dest = tmp_path / "pkg"
    dest.mkdir()
    (dest / "important.txt").write_text("do not clobber")
    with pytest.raises(ExportError, match="not empty"):
        write_export(store, run_id, dest)


# --- build flags must match what was verified ------------------------------


def test_manifest_records_the_nvcc_flags_the_kernel_was_built_with() -> None:
    import json

    store = InMemoryStore()
    run_id = _populate(store)
    store.write_json(run_id, "inputs/config.json", {"nvcc_flags": ["-O3", "--use_fast_math"]})
    manifest = json.loads(build_export(store, run_id)["ce_rms_norm_fp16/manifest.json"])
    assert manifest["nvcc_flags"] == ["-O3", "--use_fast_math"]


def test_loader_builds_with_the_recorded_nvcc_flags() -> None:
    """A package built with different flags than were measured is not the same kernel."""
    store = InMemoryStore()
    run_id = _populate(store)
    store.write_json(run_id, "inputs/config.json", {"nvcc_flags": ["-O3", "--use_fast_math"]})
    init = build_export(store, run_id)["ce_rms_norm_fp16/__init__.py"]
    assert "extra_cuda_cflags" in init
    assert "--use_fast_math" in init


def test_missing_config_falls_back_to_documented_defaults() -> None:
    import json

    store = InMemoryStore()
    run_id = _populate(store)  # no inputs/config.json written
    manifest = json.loads(build_export(store, run_id)["ce_rms_norm_fp16/manifest.json"])
    assert manifest["nvcc_flags"] == ["-O3", "--use_fast_math"]


# --- the accepted kernel is whatever the run recorded, not a guessed path ----


def _with_artifact(store: InMemoryStore, run_id: str, recorded_path: str) -> None:
    checkpoint = store.read_json(run_id, "checkpoint.json")
    checkpoint["objects"]["artifact"] = {"kernel_cu_path": recorded_path}
    store.write_json(run_id, "checkpoint.json", checkpoint)


def test_uses_the_kernel_path_the_run_actually_recorded() -> None:
    """Repair/escalation kernels live under paths no candidate list can enumerate."""
    import json

    store = InMemoryStore()
    run_id = _populate(store)
    store.write_text(run_id, "stage3_repair/attempt_01/codegen/final/kernel.cu", "// repaired\n")
    _with_artifact(store, run_id, f"<memory>/{run_id}/stage3_repair/attempt_01/codegen/final/kernel.cu")
    files = build_export(store, run_id)
    assert files["ce_rms_norm_fp16/kernel.cu"] == "// repaired\n"
    manifest = json.loads(files["ce_rms_norm_fp16/manifest.json"])
    assert manifest["source_kernel"] == "stage3_repair/attempt_01/codegen/final/kernel.cu"


def test_recorded_path_from_another_machine_still_resolves() -> None:
    """Runs get moved to Drive; the absolute path recorded at synthesis time won't match."""
    store = InMemoryStore()
    run_id = _populate(store)
    store.write_text(run_id, "stage3_repair/attempt_00/codegen/final/kernel.cu", "// moved\n")
    _with_artifact(
        store, run_id,
        f"/some/other/machine/runs/{run_id}/stage3_repair/attempt_00/codegen/final/kernel.cu",
    )
    assert build_export(store, run_id)["ce_rms_norm_fp16/kernel.cu"] == "// moved\n"


def test_recorded_path_that_does_not_exist_falls_back_to_candidates() -> None:
    store = InMemoryStore()
    run_id = _populate(store)
    _with_artifact(store, run_id, f"<memory>/{run_id}/stage9_nope/kernel.cu")
    assert build_export(store, run_id)["ce_rms_norm_fp16/kernel.cu"] == KERNEL_SRC


def test_escalated_codegen_kernel_is_a_candidate() -> None:
    import json

    store = InMemoryStore()
    run_id = _populate(store)
    store._files.pop((run_id, "stage5_polish/final/kernel.cu"))
    store.write_text(run_id, "stage2_codegen/escalated/final/kernel.cu", "// escalated\n")
    files = build_export(store, run_id)
    assert files["ce_rms_norm_fp16/kernel.cu"] == "// escalated\n"
    assert json.loads(files["ce_rms_norm_fp16/manifest.json"])["source_kernel"] == (
        "stage2_codegen/escalated/final/kernel.cu"
    )


def test_error_names_every_location_that_was_tried() -> None:
    store = InMemoryStore()
    run_id = _populate(store)
    store._files.pop((run_id, "stage5_polish/final/kernel.cu"))
    with pytest.raises(ExportError, match=re.escape("stage2_codegen/final/kernel.cu")):
        build_export(store, run_id)


def test_pyproject_declares_ninja_for_the_jit_build_path() -> None:
    """The default load path is a source build, which requires ninja."""
    store = InMemoryStore()
    run_id = _populate(store)
    data = tomllib.loads(build_export(store, run_id)["pyproject.toml"])
    deps = " ".join(data["project"]["dependencies"])
    assert "ninja" in deps


def test_loader_explains_how_to_fix_a_missing_build_toolchain() -> None:
    store = InMemoryStore()
    run_id = _populate(store)
    init = build_export(store, run_id)["ce_rms_norm_fp16/__init__.py"]
    assert "ninja" in init
