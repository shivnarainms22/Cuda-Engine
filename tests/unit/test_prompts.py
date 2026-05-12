import pytest

from cuda_engine.prompts import load_prompt


def test_load_prompt_finds_codegen_prompt() -> None:
    prompt = load_prompt("codegen")

    assert "compile_kernel" in prompt
    assert "CUDA" in prompt
    assert "cuda_engine::forward" in prompt
    assert "torch.ops.cuda_engine.forward" in prompt
    assert "TORCH_LIBRARY" in prompt
    assert "argmax" in prompt
    assert "int64" in prompt
    assert "RMSNorm" in prompt
    assert "fp32 accumulation" in prompt


def test_load_prompt_finds_polish_prompt() -> None:
    prompt = load_prompt("polish")

    assert "annotated" in prompt
    assert "tile" in prompt


def test_load_prompt_finds_perf_fix_triage_guidance() -> None:
    prompt = load_prompt("perf_fix")

    assert "__half2" in prompt
    assert "one-pass pointwise" in prompt
    assert "Do not add multi-pass reductions" in prompt


def test_load_prompt_raises_for_unknown_prompt() -> None:
    with pytest.raises(FileNotFoundError, match="Prompt not found"):
        load_prompt("missing")


def test_load_prompt_perf_fix_includes_beat_torch_compile_guidance() -> None:
    """perf_fix gives the LLM concrete levers to push past 1.0x parity."""
    prompt = load_prompt("perf_fix")

    assert "Matching torch.compile is acceptable but not the goal" in prompt
    assert "float4" in prompt
    assert "108 SMs" in prompt
    assert "#pragma unroll" in prompt
    assert "__shfl_down_sync" in prompt
