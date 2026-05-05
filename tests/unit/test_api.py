from cuda_engine import SynthesisConfig, SynthesisResult, synthesize
from cuda_engine.services.gpu.base import CompileResult
from cuda_engine.services.gpu.mocks import MockGPURunner
from cuda_engine.services.llm.base import LLMResponse
from cuda_engine.services.llm.mocks import MockLLMClient
from cuda_engine.services.store.mocks import InMemoryStore

SPEC_JSON = """{"name":"identity","target_arch":"sm_80","inputs":[{"name":"x","dtype":"fp32","shape":["N"]}],"outputs":[{"name":"out","dtype":"fp32","shape":["N"]}],"precision_tolerance":{"rtol":0.001,"atol":0.001},"optimization_priority":"balanced"}"""


def test_synthesize_returns_result_with_mocks() -> None:
    result = synthesize(
        prompt="noop",
        reference=lambda x: x,
        target="sm_80",
        config=SynthesisConfig(),
        _llm=MockLLMClient(
            responses=[
                SPEC_JSON,
                LLMResponse(
                    text="```cuda\nextern code\n```",
                    model="mock",
                    tool_calls=[
                        {
                            "name": "compile_kernel",
                            "input": {"src": "extern code", "target_arch": "sm_80"},
                        }
                    ],
                )
            ]
        ),
        _gpu=MockGPURunner(compile_results=[CompileResult(ok=True, log="ok")]),
        _store=InMemoryStore(),
    )

    assert isinstance(result, SynthesisResult)
    assert result.passed
