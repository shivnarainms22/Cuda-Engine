"""Tests that --resume / resume_run_id is threaded from CLI -> api -> orchestrator."""
from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from cuda_engine.cli import app


def test_synthesize_forwards_resume_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeOrchestrator:
        def __init__(self, **_: Any) -> None:
            pass

        def run(self, **kwargs: Any) -> str:
            captured.update(kwargs)
            return "RESULT"

    monkeypatch.setattr("cuda_engine.api.Orchestrator", _FakeOrchestrator)
    from cuda_engine import api

    out = api.synthesize(
        "p", lambda x: x, resume_run_id="r1", _llm=object(), _gpu=object(), _store=object()
    )
    assert out == "RESULT"
    assert captured["resume_run_id"] == "r1"


def test_synthesize_resume_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeOrchestrator:
        def __init__(self, **_: Any) -> None:
            pass

        def run(self, **kwargs: Any) -> str:
            captured.update(kwargs)
            return "RESULT"

    monkeypatch.setattr("cuda_engine.api.Orchestrator", _FakeOrchestrator)
    from cuda_engine import api

    api.synthesize("p", lambda x: x, _llm=object(), _gpu=object(), _store=object())
    assert captured["resume_run_id"] is None


def test_cli_synthesize_exposes_resume_option() -> None:
    result = CliRunner().invoke(app, ["synthesize", "--help"])
    assert result.exit_code == 0
    assert "resume" in result.output
