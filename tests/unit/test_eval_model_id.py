"""Tests for per-provider eval routing (--model-id) + provider/model_id columns."""
from __future__ import annotations

import json
from pathlib import Path

from cuda_engine.cli import _eval_config
from evals.runner import CSV_COLUMNS, EvalRow, _row_from_json, _row_to_csv, _row_to_json


def test_eval_config_default_is_anthropic() -> None:
    cfg = _eval_config(None)
    assert cfg.stage_models.interview == "anthropic:claude-sonnet-4-6"


def test_eval_config_routes_every_stage_to_model_id() -> None:
    cfg = _eval_config("openai:gpt-4o")
    sm = cfg.stage_models
    assert sm.interview == "openai:gpt-4o"
    assert sm.codegen == "openai:gpt-4o"
    assert sm.correctness == "openai:gpt-4o"
    assert sm.performance == "openai:gpt-4o"
    assert sm.polish == "openai:gpt-4o"


def test_csv_columns_include_provider_and_model_id() -> None:
    assert "provider" in CSV_COLUMNS
    assert "model_id" in CSV_COLUMNS


def _row(**kw: object) -> EvalRow:
    base = dict(
        kernel="k",
        passed=True,
        run_id="r",
        failed_stage=None,
        failure_reason="",
        speedup_vs_torch_compile=1.2,
        speedup_vs_reference=2.0,
        below_target=False,
        artifacts_dir="/a",
        provider="openai",
        model_id="openai:gpt-4o",
    )
    base.update(kw)
    return EvalRow(**base)  # type: ignore[arg-type]


def test_row_to_csv_emits_provider_and_model_id() -> None:
    csv_row = _row_to_csv(_row())
    assert csv_row["provider"] == "openai"
    assert csv_row["model_id"] == "openai:gpt-4o"


def test_row_json_roundtrip_preserves_provider_and_model_id(tmp_path: Path) -> None:
    path = tmp_path / "k.json"
    path.write_text(json.dumps(_row_to_json(_row())), encoding="utf-8")
    restored = _row_from_json(path)
    assert restored.provider == "openai"
    assert restored.model_id == "openai:gpt-4o"


def test_row_from_legacy_json_defaults_provider_blank(tmp_path: Path) -> None:
    """Old result JSONs without provider/model_id must still load (back-compat)."""
    legacy = {
        "kernel": "k",
        "passed": True,
        "run_id": "r",
        "failed_stage": None,
        "failure_reason": "",
        "speedup_vs_torch_compile": 1.0,
        "speedup_vs_reference": 1.0,
        "below_target": False,
        "artifacts_dir": "/a",
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    restored = _row_from_json(path)
    assert restored.provider == ""
    assert restored.model_id == ""
