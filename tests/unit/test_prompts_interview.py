from cuda_engine.prompts import load_prompt


def test_load_prompt_finds_interview_prompt() -> None:
    prompt = load_prompt("interview")

    assert "KernelSpec" in prompt
    assert "JSON" in prompt
    assert "argmax" in prompt
    assert "int64" in prompt
