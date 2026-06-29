"""Tests for the pure payload-translation layer (no SDK imports)."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from cuda_engine.services.llm.base import ToolSpec
from cuda_engine.services.llm.translate import (
    from_gemini_response,
    from_openai_response,
    to_gemini,
    to_openai_messages,
    to_openai_tools,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _ns(**kwargs: Any) -> SimpleNamespace:
    """Recursive SimpleNamespace builder for clean fixture construction."""
    result = SimpleNamespace()
    for k, v in kwargs.items():
        if isinstance(v, dict):
            setattr(result, k, _ns(**v))
        else:
            setattr(result, k, v)
    return result


def _oai_text_resp(text: str = "hello", tokens_in: int = 10, tokens_out: int = 5) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=text,
                    tool_calls=None,
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=tokens_in,
            completion_tokens=tokens_out,
            prompt_tokens_details=None,
        ),
    )


def _oai_tool_resp(name: str = "search", args: dict | None = None, cached: int = 3) -> Any:
    args = args or {"query": "test"}
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_abc",
                            function=SimpleNamespace(
                                name=name,
                                arguments=json.dumps(args),
                            ),
                        )
                    ],
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=20,
            completion_tokens=8,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        ),
    )


def _gemini_text_resp(text: str = "gemini answer") -> Any:
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text=text, function_call=None)]
                )
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=7,
        ),
    )


def _gemini_tool_resp(name: str = "lookup", args: dict | None = None) -> Any:
    args = args or {"key": "value"}
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            text=None,
                            function_call=SimpleNamespace(name=name, args=args),
                        )
                    ]
                )
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=15,
            candidates_token_count=3,
        ),
    )


# ---------------------------------------------------------------------------
# to_openai_messages
# ---------------------------------------------------------------------------


def test_to_openai_messages_system_text_becomes_leading_system_role() -> None:
    system = [{"type": "text", "text": "You are helpful."}]
    messages = [{"role": "user", "content": "hi"}]
    result = to_openai_messages(system, messages)
    assert result[0] == {"role": "system", "content": "You are helpful."}


def test_to_openai_messages_cache_control_dropped_from_system() -> None:
    system = [
        {"type": "text", "text": "Sys prompt.", "cache_control": {"type": "ephemeral"}}
    ]
    result = to_openai_messages(system, [])
    assert result[0]["content"] == "Sys prompt."
    assert "cache_control" not in result[0]


def test_to_openai_messages_multiple_system_blocks_concatenated() -> None:
    system = [
        {"type": "text", "text": "Part one."},
        {"type": "text", "text": "Part two."},
    ]
    result = to_openai_messages(system, [])
    assert result[0]["content"] == "Part one.\nPart two."


def test_to_openai_messages_empty_system_omits_system_role() -> None:
    result = to_openai_messages([], [{"role": "user", "content": "hi"}])
    assert result[0]["role"] == "user"


def test_to_openai_messages_user_string_content_preserved() -> None:
    result = to_openai_messages([], [{"role": "user", "content": "hello"}])
    assert result == [{"role": "user", "content": "hello"}]


def test_to_openai_messages_assistant_string_content_preserved() -> None:
    result = to_openai_messages(
        [], [{"role": "assistant", "content": "I can help."}]
    )
    assert result == [{"role": "assistant", "content": "I can help."}]


def test_to_openai_messages_assistant_tool_use_block() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me search."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "search",
                    "input": {"query": "weather"},
                },
            ],
        }
    ]
    result = to_openai_messages([], messages)
    assert len(result) == 1
    msg = result[0]
    assert msg["role"] == "assistant"
    assert msg["content"] == "Let me search."
    assert len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc["id"] == "toolu_1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "search"
    assert json.loads(tc["function"]["arguments"]) == {"query": "weather"}


def test_to_openai_messages_assistant_tool_use_no_text_has_none_content() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_2", "name": "fn", "input": {}}
            ],
        }
    ]
    result = to_openai_messages([], messages)
    assert result[0]["content"] is None
    assert len(result[0]["tool_calls"]) == 1


def test_to_openai_messages_user_tool_result_block() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "search results here",
                }
            ],
        }
    ]
    result = to_openai_messages([], messages)
    assert len(result) == 1
    assert result[0] == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "content": "search results here",
    }


def test_to_openai_messages_user_tool_result_with_block_content() -> None:
    """tool_result whose content is a list of blocks — extract text."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_3",
                    "content": [{"type": "text", "text": "block text"}],
                }
            ],
        }
    ]
    result = to_openai_messages([], messages)
    assert result[0]["content"] == "block text"


def test_to_openai_messages_full_round_trip() -> None:
    """system → user → assistant with tool → user tool_result."""
    system = [{"type": "text", "text": "You are a bot."}]
    messages = [
        {"role": "user", "content": "Search for X"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_x",
                    "name": "search",
                    "input": {"q": "X"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_x", "content": "found X"}
            ],
        },
    ]
    result = to_openai_messages(system, messages)
    roles = [m["role"] for m in result]
    assert roles == ["system", "user", "assistant", "tool"]


# ---------------------------------------------------------------------------
# to_openai_tools
# ---------------------------------------------------------------------------


def test_to_openai_tools_none_returns_none() -> None:
    assert to_openai_tools(None) is None


def test_to_openai_tools_empty_list_returns_empty_list() -> None:
    assert to_openai_tools([]) == []


def test_to_openai_tools_converts_toolspec() -> None:
    tools = [
        ToolSpec(
            name="calculator",
            description="Does math",
            input_schema={"type": "object", "properties": {"expr": {"type": "string"}}},
        )
    ]
    result = to_openai_tools(tools)
    assert result is not None
    assert len(result) == 1
    assert result[0]["type"] == "function"
    fn = result[0]["function"]
    assert fn["name"] == "calculator"
    assert fn["description"] == "Does math"
    assert fn["parameters"] == tools[0].input_schema


# ---------------------------------------------------------------------------
# to_gemini
# ---------------------------------------------------------------------------


def test_to_gemini_system_instruction_from_text_blocks() -> None:
    system = [{"type": "text", "text": "Be concise."}]
    result = to_gemini(system, [], None)
    assert result["system_instruction"] == "Be concise."


def test_to_gemini_cache_control_stripped_from_system() -> None:
    system = [
        {"type": "text", "text": "Prompt.", "cache_control": {"type": "ephemeral"}}
    ]
    result = to_gemini(system, [], None)
    assert result["system_instruction"] == "Prompt."


def test_to_gemini_user_text_message() -> None:
    result = to_gemini([], [{"role": "user", "content": "hello"}], None)
    assert result["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]


def test_to_gemini_assistant_maps_to_model_role() -> None:
    result = to_gemini([], [{"role": "assistant", "content": "hi there"}], None)
    assert result["contents"][0]["role"] == "model"


def test_to_gemini_assistant_tool_use_becomes_function_call() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "search",
                    "input": {"q": "cats"},
                }
            ],
        }
    ]
    result = to_gemini([], messages, None)
    part = result["contents"][0]["parts"][0]
    assert part == {"function_call": {"name": "search", "args": {"q": "cats"}}}


def test_to_gemini_tool_result_becomes_function_response_with_name_lookup() -> None:
    """Name is resolved by cross-referencing the preceding tool_use id."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {}}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "results"}
            ],
        },
    ]
    result = to_gemini([], messages, None)
    user_part = result["contents"][1]["parts"][0]
    assert user_part == {
        "function_response": {
            "name": "search",
            "response": {"result": "results"},
        }
    }


def test_to_gemini_tools_none_yields_empty_list() -> None:
    result = to_gemini([], [], None)
    assert result["tools"] == []


def test_to_gemini_tools_produces_function_declarations() -> None:
    tools = [
        ToolSpec(
            name="weather",
            description="Get weather",
            input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
        )
    ]
    result = to_gemini([], [], tools)
    assert len(result["tools"]) == 1
    decls = result["tools"][0]["function_declarations"]
    assert decls[0]["name"] == "weather"
    assert decls[0]["description"] == "Get weather"
    assert decls[0]["parameters"] == tools[0].input_schema


def test_to_gemini_multiple_tools_in_single_declaration_block() -> None:
    tools = [
        ToolSpec(name="fn1", description="d1", input_schema={}),
        ToolSpec(name="fn2", description="d2", input_schema={}),
    ]
    result = to_gemini([], [], tools)
    # All tools in ONE function_declarations block
    assert len(result["tools"]) == 1
    assert len(result["tools"][0]["function_declarations"]) == 2


# ---------------------------------------------------------------------------
# from_openai_response
# ---------------------------------------------------------------------------


def test_from_openai_response_text_only() -> None:
    resp = _oai_text_resp("Hello!")
    parsed = from_openai_response(resp)
    assert parsed["text"] == "Hello!"
    assert parsed["tool_calls"] == []
    assert parsed["tokens_in"] == 10
    assert parsed["tokens_out"] == 5
    assert parsed["cache_read_tokens"] == 0


def test_from_openai_response_none_content_becomes_empty_string() -> None:
    # Build manually: content=None
    r = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=None))],
        usage=SimpleNamespace(
            prompt_tokens=0, completion_tokens=0, prompt_tokens_details=None
        ),
    )
    parsed = from_openai_response(r)
    assert parsed["text"] == ""


def test_from_openai_response_tool_calls_parsed() -> None:
    resp = _oai_tool_resp("search", {"q": "hello"}, cached=0)
    parsed = from_openai_response(resp)
    assert parsed["text"] == ""
    assert len(parsed["tool_calls"]) == 1
    tc = parsed["tool_calls"][0]
    assert tc["name"] == "search"
    assert tc["input"] == {"q": "hello"}


def test_from_openai_response_cached_tokens_read() -> None:
    resp = _oai_tool_resp(cached=7)
    parsed = from_openai_response(resp)
    assert parsed["cache_read_tokens"] == 7


def test_from_openai_response_no_prompt_tokens_details() -> None:
    resp = _oai_text_resp()
    assert resp.usage.prompt_tokens_details is None
    parsed = from_openai_response(resp)
    assert parsed["cache_read_tokens"] == 0


# ---------------------------------------------------------------------------
# from_gemini_response
# ---------------------------------------------------------------------------


def test_from_gemini_response_text_only() -> None:
    resp = _gemini_text_resp("Gemini says hi")
    parsed = from_gemini_response(resp)
    assert parsed["text"] == "Gemini says hi"
    assert parsed["tool_calls"] == []
    assert parsed["tokens_in"] == 12
    assert parsed["tokens_out"] == 7


def test_from_gemini_response_function_call_part() -> None:
    resp = _gemini_tool_resp("lookup", {"id": "42"})
    parsed = from_gemini_response(resp)
    assert parsed["text"] == ""
    assert len(parsed["tool_calls"]) == 1
    tc = parsed["tool_calls"][0]
    assert tc["name"] == "lookup"
    assert tc["input"] == {"id": "42"}


def test_from_gemini_response_returns_cache_read_tokens_key() -> None:
    resp = _gemini_text_resp()
    parsed = from_gemini_response(resp)
    assert "cache_read_tokens" in parsed
