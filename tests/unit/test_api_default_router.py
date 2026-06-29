"""Regression tests: the default synthesize() path must inject an LLMRouter.

The orchestrator passes provider-prefixed model ids from cfg.stage_models
(e.g. "anthropic:claude-sonnet-4-6"). A raw AnthropicClient would forward that
prefixed string to the API as the model name and fail, so the default client
MUST be the router that strips the prefix and dispatches by provider.
"""
from __future__ import annotations

import pytest

from cuda_engine.config import SynthesisConfig
from cuda_engine.orchestrator import build_router
from cuda_engine.services.llm.mocks import MockLLMClient
from cuda_engine.services.llm.router import LLMRouter


def test_build_router_default_registers_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    router = build_router(SynthesisConfig())
    assert isinstance(router, LLMRouter)


def test_build_router_adds_openai_and_gemini_only_when_keys_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    router = build_router(SynthesisConfig())
    # openai registered (key present), gemini not (key absent)
    assert "openai" in router._providers
    assert "gemini" not in router._providers


def test_default_stage_model_ids_dispatch_to_anthropic_with_stripped_prefix() -> None:
    """The default config's stage ids must route to anthropic with the prefix
    stripped — proving the orchestrator's stage_models ids work through a router
    (and would NOT work if fed to a raw AnthropicClient)."""
    cfg = SynthesisConfig()
    mock = MockLLMClient(["ok"])
    router = LLMRouter({"anthropic": mock})
    router.complete(system=[], messages=[], model=cfg.stage_models.interview)
    assert mock.calls[0]["model"] == "claude-sonnet-4-6"


def test_synthesize_default_llm_is_a_router(monkeypatch: pytest.MonkeyPatch) -> None:
    """synthesize() with no injected _llm must build a router, not a raw client."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    captured: dict[str, object] = {}

    class _StopHere(RuntimeError):
        pass

    def _fake_orchestrator(*, llm: object, gpu: object, store: object, cfg: object) -> object:
        captured["llm"] = llm
        raise _StopHere

    monkeypatch.setattr("cuda_engine.api.Orchestrator", _fake_orchestrator)
    from cuda_engine import api

    with pytest.raises(_StopHere):
        api.synthesize("p", lambda x: x, _gpu=object(), _store=object())
    assert isinstance(captured["llm"], LLMRouter)
