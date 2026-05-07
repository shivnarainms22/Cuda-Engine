import pytest
from pydantic import ValidationError

from cuda_engine.services.llm.mocks import MockLLMClient
from cuda_engine.services.store.mocks import InMemoryStore
from cuda_engine.stages.base import StructuralStageError
from cuda_engine.stages.interview import Stage1Interview, _introspect_reference, _parse_kernel_spec


def test_stage1_interview_parses_kernel_spec_from_fenced_json() -> None:
    llm = MockLLMClient(
        responses=[
            """```json
{
  "name": "vector_add",
  "target_arch": "sm_80",
  "inputs": [
    {"name": "x", "dtype": "fp32", "shape": ["N"]},
    {"name": "y", "dtype": "fp32", "shape": ["N"]}
  ],
  "outputs": [{"name": "out", "dtype": "fp32", "shape": ["N"]}],
  "precision_tolerance": {"rtol": 1e-5, "atol": 1e-6},
  "optimization_priority": "throughput"
}
```"""
        ]
    )
    store = InMemoryStore()
    stage = Stage1Interview(llm=llm, store=store)

    spec = stage.run(
        prompt="vector add",
        reference=lambda x, y: x + y,
        target_arch="sm_80",
        run_id="run123",
    )

    assert spec.name == "vector_add"
    assert spec.inputs[0].name == "x"
    assert spec.optimization_priority == "throughput"
    assert llm.calls[0]["model"] == "claude-sonnet-4-6"
    assert ("run123", "stage1_interview/kernel_spec.json") in store._files


def test_stage1_interview_rejects_invalid_json() -> None:
    stage = Stage1Interview(llm=MockLLMClient(["not json"]), store=InMemoryStore())

    with pytest.raises(StructuralStageError, match="KernelSpec JSON"):
        stage.run(prompt="bad", reference=lambda x: x, target_arch="sm_80", run_id="run123")


def test_stage1_interview_normalizes_contiguous_layout_hint() -> None:
    spec = _parse_kernel_spec(
        """```json
{
  "name": "vector_add",
  "target_arch": "sm_80",
  "inputs": [{"name": "x", "dtype": "fp32", "shape": ["N"], "layout_hint": "contiguous"}],
  "outputs": [{"name": "out", "dtype": "fp32", "shape": ["N"], "layout_hint": "contiguous"}],
  "precision_tolerance": {"rtol": 1e-5, "atol": 1e-6},
  "optimization_priority": "throughput"
}
```"""
    )

    assert spec.inputs[0].layout_hint == "row_major"
    assert spec.outputs[0].layout_hint == "row_major"


def test_stage1_interview_returns_frozen_spec() -> None:
    stage = Stage1Interview(
        llm=MockLLMClient(
            [
                """{"name":"identity","target_arch":"sm_80","inputs":[{"name":"x","dtype":"fp32","shape":["N"]}],"outputs":[{"name":"out","dtype":"fp32","shape":["N"]}],"precision_tolerance":{"rtol":0.001,"atol":0.001},"optimization_priority":"balanced"}"""
            ]
        ),
        store=InMemoryStore(),
    )

    spec = stage.run(prompt="identity", reference=lambda x: x, target_arch="sm_80", run_id="run123")

    with pytest.raises(ValidationError):
        spec.name = "mutated"


def test_introspect_reference_captures_signature() -> None:
    def reference(x, y, scale=1.0):
        return (x + y) * scale

    info = _introspect_reference(reference)

    assert info["callable_name"] == "reference"
    assert info["parameters"] == ["x", "y", "scale"]
    assert info["error"] is None
