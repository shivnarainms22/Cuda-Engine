import inspect
import json
import re
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from cuda_engine.models import KernelSpec
from cuda_engine.prompts import load_prompt
from cuda_engine.stages.base import Stage, StructuralStageError


class Stage1Interview(Stage):
    name = "interview"

    def run(
        self,
        *,
        prompt: str,
        reference: Callable[..., Any],
        target_arch: str,
        run_id: str,
    ) -> KernelSpec:
        if self.llm is None or self.store is None:
            raise RuntimeError("Stage1Interview requires llm and store services")

        metadata = _introspect_reference(reference)
        user_message = {
            "role": "user",
            "content": (
                "Create a KernelSpec for this request.\n\n"
                f"Prompt:\n{prompt}\n\n"
                f"Target architecture: {target_arch}\n\n"
                f"Reference metadata:\n{json.dumps(metadata, indent=2)}"
            ),
        }
        system = [
            {
                "type": "text",
                "text": load_prompt("interview"),
                "cache_control": {"type": "ephemeral"},
            }
        ]
        self.store.write_text(run_id, "stage1_interview/prompt_to_llm.md", user_message["content"])

        response = self.llm.complete(
            system=system,
            messages=[user_message],
            tools=None,
            model="claude-sonnet-4-6",
        )
        self.store.write_text(run_id, "stage1_interview/llm_response.md", response.text)
        spec = _parse_kernel_spec(response.text)
        self.store.write_json(run_id, "stage1_interview/kernel_spec.json", spec)
        return spec


def _introspect_reference(reference: Callable[..., Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(reference)
        return {
            "callable_name": getattr(reference, "__name__", type(reference).__name__),
            "parameters": list(signature.parameters),
            "signature": str(signature),
            "error": None,
        }
    except (TypeError, ValueError) as exc:
        return {
            "callable_name": getattr(reference, "__name__", type(reference).__name__),
            "parameters": [],
            "signature": None,
            "error": str(exc),
        }


def _parse_kernel_spec(text: str) -> KernelSpec:
    payload = _extract_json(text)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise StructuralStageError(f"KernelSpec JSON could not be decoded: {exc}") from exc
    _normalize_kernel_spec_dict(data)
    try:
        return KernelSpec.model_validate(data)
    except ValidationError as exc:
        raise StructuralStageError(f"KernelSpec JSON failed validation: {exc}") from exc


def _extract_json(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    raise StructuralStageError("KernelSpec JSON was not found in interview response")


def _normalize_kernel_spec_dict(data: object) -> None:
    if not isinstance(data, dict):
        return
    for section in ("inputs", "outputs"):
        args = data.get(section)
        if not isinstance(args, list):
            continue
        for arg in args:
            if not isinstance(arg, dict):
                continue
            layout_hint = arg.get("layout_hint")
            if layout_hint in {"contiguous", "c_contiguous", "strided_contiguous"}:
                arg["layout_hint"] = "row_major"
