import json

from cuda_engine.services.llm.tools import (
    ALL_TOOLS,
    COMPILE_KERNEL,
    NSIGHT_PROFILE,
    RUN_CORRECTNESS,
)


def test_tool_schemas_are_json_serializable() -> None:
    for tool in ALL_TOOLS:
        json.dumps(tool.model_dump(mode="json"))
        assert tool.input_schema["type"] == "object"
        assert "properties" in tool.input_schema


def test_compile_kernel_schema_requires_src() -> None:
    assert COMPILE_KERNEL.name == "compile_kernel"
    assert COMPILE_KERNEL.input_schema["required"] == ["src", "target_arch"]
    assert "src" in COMPILE_KERNEL.input_schema["properties"]


def test_correctness_and_profile_tools_are_named() -> None:
    assert RUN_CORRECTNESS.name == "run_correctness"
    assert NSIGHT_PROFILE.name == "nsight_profile"
