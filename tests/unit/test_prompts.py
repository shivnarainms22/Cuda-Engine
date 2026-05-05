import pytest

from cuda_engine.prompts import load_prompt


def test_load_prompt_finds_codegen_prompt() -> None:
    prompt = load_prompt("codegen")

    assert "compile_kernel" in prompt
    assert "CUDA" in prompt
    assert "cuda_engine::forward" in prompt
    assert "torch.ops.cuda_engine.forward" in prompt
    assert "TORCH_LIBRARY" in prompt


def test_load_prompt_finds_polish_prompt() -> None:
    prompt = load_prompt("polish")

    assert "annotated" in prompt
    assert "tile" in prompt


def test_load_prompt_raises_for_unknown_prompt() -> None:
    with pytest.raises(FileNotFoundError, match="Prompt not found"):
        load_prompt("missing")
