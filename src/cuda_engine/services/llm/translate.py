"""Pure payload translation between Anthropic-native and OpenAI / Gemini formats.

No SDK imports — stdlib + pydantic only.
"""
from __future__ import annotations

import json
from typing import Any

from cuda_engine.services.llm.base import ToolSpec

# ---------------------------------------------------------------------------
# → OpenAI
# ---------------------------------------------------------------------------


def to_openai_messages(
    system: list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Anthropic-native system + messages to the OpenAI messages list."""
    result: list[dict[str, Any]] = []

    # System: concat all text blocks (drop cache_control)
    system_text_parts = [
        block["text"]
        for block in system
        if block.get("type") == "text" and block.get("text")
    ]
    if system_text_parts:
        result.append({"role": "system", "content": "\n".join(system_text_parts)})

    for msg in messages:
        role: str = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            result.append({"role": role, "content": content})
        elif isinstance(content, list):
            if role == "assistant":
                result.extend(_assistant_blocks_to_oai(content))
            else:
                result.extend(_user_blocks_to_oai(content))

    return result


def _assistant_blocks_to_oai(
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in blocks:
        btype = block.get("type")
        if btype == "text" and block.get("text"):
            text_parts.append(block["text"])
        elif btype == "tool_use":
            tool_calls.append(
                {
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block.get("input", {})),
                    },
                }
            )

    text: str | None = "\n".join(text_parts) or None
    msg: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return [msg]


def _user_blocks_to_oai(
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    text_parts: list[str] = []
    tool_messages: list[dict[str, Any]] = []

    for block in blocks:
        btype = block.get("type")
        if btype == "text" and block.get("text"):
            text_parts.append(block["text"])
        elif btype == "tool_result":
            raw = block.get("content", "")
            if isinstance(raw, list):
                raw = "\n".join(
                    b.get("text", "") for b in raw if b.get("type") == "text"
                )
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": block["tool_use_id"],
                    "content": str(raw),
                }
            )

    result: list[dict[str, Any]] = []
    if text_parts:
        result.append({"role": "user", "content": "\n".join(text_parts)})
    result.extend(tool_messages)
    return result


def to_openai_tools(
    tools: list[ToolSpec] | None,
) -> list[dict[str, Any]] | None:
    """Convert ToolSpec list to OpenAI function-calling format. None stays None."""
    if tools is None:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


# ---------------------------------------------------------------------------
# → Gemini
# ---------------------------------------------------------------------------


def to_gemini(
    system: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    tools: list[ToolSpec] | None,
) -> dict[str, Any]:
    """Convert Anthropic-native payloads to the google-genai generate_content shape."""
    system_instruction = "\n".join(
        block["text"]
        for block in system
        if block.get("type") == "text" and block.get("text")
    )

    # Pre-pass: build tool_use_id → name map for tool_result name resolution
    id_to_name: dict[str, str] = {}
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_use":
                    id_to_name[block["id"]] = block["name"]

    contents: list[dict[str, Any]] = []
    for msg in messages:
        role: str = msg["role"]
        gemini_role = "model" if role == "assistant" else "user"
        content = msg["content"]

        parts: list[dict[str, Any]] = []
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    parts.append({"text": block.get("text", "")})
                elif btype == "tool_use":
                    parts.append(
                        {
                            "function_call": {
                                "name": block["name"],
                                "args": block.get("input", {}),
                            }
                        }
                    )
                elif btype == "tool_result":
                    uid = block["tool_use_id"]
                    name = id_to_name.get(uid, uid)
                    raw = block.get("content", "")
                    if isinstance(raw, list):
                        raw = "\n".join(
                            b.get("text", "") for b in raw if b.get("type") == "text"
                        )
                    parts.append(
                        {
                            "function_response": {
                                "name": name,
                                "response": {"result": raw},
                            }
                        }
                    )

        if parts:
            contents.append({"role": gemini_role, "parts": parts})

    gemini_tools: list[dict[str, Any]] = []
    if tools:
        gemini_tools = [
            {
                "function_declarations": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    }
                    for t in tools
                ]
            }
        ]

    return {
        "system_instruction": system_instruction,
        "contents": contents,
        "tools": gemini_tools,
    }


# ---------------------------------------------------------------------------
# ← OpenAI
# ---------------------------------------------------------------------------


def from_openai_response(resp: Any) -> dict[str, Any]:
    """Parse an OpenAI ChatCompletion response (or a SimpleNamespace mirror) into a
    normalised dict compatible with LLMResponse fields."""
    msg = resp.choices[0].message
    text: str = msg.content or ""

    tool_calls: list[dict[str, Any]] = []
    for tc in getattr(msg, "tool_calls", None) or []:
        tool_calls.append(
            {
                "name": tc.function.name,
                "input": json.loads(tc.function.arguments),
            }
        )

    usage = resp.usage
    tokens_in: int = int(getattr(usage, "prompt_tokens", 0) or 0)
    tokens_out: int = int(getattr(usage, "completion_tokens", 0) or 0)

    details = getattr(usage, "prompt_tokens_details", None)
    cache_read_tokens: int = 0
    if details is not None:
        cache_read_tokens = int(getattr(details, "cached_tokens", 0) or 0)

    return {
        "text": text,
        "tool_calls": tool_calls,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cache_read_tokens": cache_read_tokens,
    }


# ---------------------------------------------------------------------------
# ← Gemini
# ---------------------------------------------------------------------------


def from_gemini_response(resp: Any) -> dict[str, Any]:
    """Parse a Gemini GenerateContentResponse (or SimpleNamespace mirror) into a
    normalised dict compatible with LLMResponse fields."""
    parts = resp.candidates[0].content.parts
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for part in parts:
        txt = getattr(part, "text", None)
        if txt:
            text_parts.append(txt)
        fc = getattr(part, "function_call", None)
        if fc is not None:
            tool_calls.append(
                {
                    "name": fc.name,
                    "input": dict(fc.args),
                }
            )

    usage = resp.usage_metadata
    tokens_in: int = int(getattr(usage, "prompt_token_count", 0) or 0)
    tokens_out: int = int(getattr(usage, "candidates_token_count", 0) or 0)

    return {
        "text": "\n".join(text_parts),
        "tool_calls": tool_calls,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cache_read_tokens": 0,
    }
