"""Tests for per-stage LLM routing via LLMRouter inside Orchestrator."""
from pathlib import Path

import pytest

from cuda_engine.config import RetryBudgets, StageModels, SynthesisConfig
from cuda_engine.orchestrator import Orchestrator, build_router
from cuda_engine.services.gpu.base import CompileResult, RunResult
from cuda_engine.services.gpu.mocks import MockGPURunner
from cuda_engine.services.llm.base import LLMResponse
from cuda_engine.services.llm.mocks import MockLLMClient
from cuda_engine.services.llm.router import LLMRouter
from cuda_engine.services.store.mocks import InMemoryStore

SPEC_JSON = '{"name":"identity","target_arch":"sm_80","inputs":[{"name":"x","dtype":"fp32","shape":["N"]}],"outputs":[{"name":"out","dtype":"fp32","shape":["N"]}],"precision_tolerance":{"rtol":0.001,"atol":0.001},"optimization_priority":"balanced"}'

SHAPE_SIZES = (0, 1, 127, 128, 1024, 4097)


def _simple_compile_response(src: str = "extern code", model: str = "mock") -> LLMResponse:
    return LLMResponse(
        text=f"```cuda\n{src}\n```",
        model=model,
        tool_calls=[{"name": "compile_kernel", "input": {"src": src, "target_arch": "sm_80"}}],
    )


def test_interview_routes_to_openai_mock() -> None:
    """When stage_models.interview='openai:gpt-x', interview call lands in openai mock."""
    torch = __import__("torch")
    store = InMemoryStore()

    anthropic_mock = MockLLMClient(
        responses=[
            _simple_compile_response("extern code", "mock-anthropic"),  # codegen
            "```cuda\n// annotated\nextern code\n```",                   # polish
        ]
    )
    openai_mock = MockLLMClient(
        responses=[
            LLMResponse(
                text=SPEC_JSON,
                model="gpt-x",
                provider="openai",
                tokens_in=10,
                tokens_out=10,
            ),
        ]
    )

    router = LLMRouter(providers={"anthropic": anthropic_mock, "openai": openai_mock})

    orchestrator = Orchestrator(
        llm=router,
        gpu=MockGPURunner(
            compile_results=[CompileResult(ok=True, so_path=Path("kernel.so"), log="ok")],
            run_results=[
                RunResult(ok=True, output_tensors=[torch.arange(size, dtype=torch.float32)])
                for size in SHAPE_SIZES
            ],
        ),
        store=store,
        cfg=SynthesisConfig(
            stage_models=StageModels(interview="openai:gpt-x"),
            retry_budgets=RetryBudgets(performance=0),
            performance_shape_n=256,
            benchmark_warmup_iterations=2,
            benchmark_timed_iterations=3,
        ),
    )

    result = orchestrator.run(prompt="noop", reference=lambda x: x, target="sm_80")

    assert result.passed is True

    # The interview call went to the openai mock (prefix-stripped model)
    assert len(openai_mock.calls) == 1
    assert openai_mock.calls[0]["model"] == "gpt-x"
    assert anthropic_mock.calls[0]["model"] != "gpt-x"

    # StageTrace.provider reflects openai
    interview_trace = next(t for t in result.report.stage_traces if t.stage_name == "interview")
    assert interview_trace.provider == "openai"


def test_stage_trace_has_provider_and_degraded_fields() -> None:
    """StageTrace now has provider and degraded fields with sensible defaults."""
    from cuda_engine.models.reports import StageTrace

    trace = StageTrace(
        stage_name="test",
        attempts=1,
        succeeded=True,
        model_used="claude-sonnet-4-6",
    )
    assert trace.provider == ""
    assert trace.degraded == []


def test_build_router_returns_llm_router(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_router always constructs and returns an LLMRouter with at least anthropic."""
    # Patch AnthropicClient so it doesn't need a real API key / SDK call
    import cuda_engine.orchestrator as orch_mod
    from cuda_engine.services.llm.mocks import MockLLMClient

    monkeypatch.setattr(
        orch_mod,
        "AnthropicClient",
        lambda cfg: MockLLMClient(responses=[]),
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    router = build_router(SynthesisConfig())

    assert isinstance(router, LLMRouter)


def test_build_router_registers_openai_when_key_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_router includes openai provider only when OPENAI_API_KEY is set."""
    import cuda_engine.orchestrator as orch_mod
    from cuda_engine.services.llm.mocks import MockLLMClient

    monkeypatch.setattr(orch_mod, "AnthropicClient", lambda cfg: MockLLMClient(responses=[]))
    monkeypatch.setattr(orch_mod, "OpenAIClient", lambda: MockLLMClient(responses=["ok"]))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    router = build_router(SynthesisConfig())

    # routing an openai: model should succeed (no KeyError)
    response = router.complete(system=[], messages=[], model="openai:gpt-4")  # type: ignore[call-arg]
    assert response.text == "ok"


def test_build_router_excludes_openai_when_key_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_router omits openai provider when OPENAI_API_KEY is absent."""
    import cuda_engine.orchestrator as orch_mod
    from cuda_engine.services.llm.mocks import MockLLMClient

    monkeypatch.setattr(orch_mod, "AnthropicClient", lambda cfg: MockLLMClient(responses=[]))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    router = build_router(SynthesisConfig())

    with pytest.raises(KeyError):
        router.complete(system=[], messages=[], model="openai:gpt-4")  # type: ignore[call-arg]
