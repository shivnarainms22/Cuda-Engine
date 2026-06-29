from cuda_engine.config import StageModels, SynthesisConfig
from cuda_engine.services.llm.router import parse_model_id

_SONNET_ID = "anthropic:claude-sonnet-4-6"


def test_stage_models_defaults_are_anthropic_sonnet() -> None:
    sm = StageModels()

    assert sm.interview == _SONNET_ID
    assert sm.codegen == _SONNET_ID
    assert sm.correctness == _SONNET_ID
    assert sm.performance == _SONNET_ID
    assert sm.polish == _SONNET_ID


def test_stage_models_defaults_parse_to_anthropic_provider() -> None:
    sm = StageModels()

    for field_value in (sm.interview, sm.codegen, sm.correctness, sm.performance, sm.polish):
        provider, bare = parse_model_id(field_value)
        assert provider == "anthropic", f"expected provider 'anthropic', got {provider!r} for {field_value!r}"
        assert bare == "claude-sonnet-4-6"


def test_synthesis_config_has_stage_models() -> None:
    cfg = SynthesisConfig()

    assert isinstance(cfg.stage_models, StageModels)


def test_synthesis_config_stage_models_defaults_to_factory() -> None:
    cfg = SynthesisConfig()

    assert cfg.stage_models.interview == _SONNET_ID
    assert cfg.stage_models.polish == _SONNET_ID


def test_stage_models_is_frozen() -> None:
    import pytest

    sm = StageModels()
    with pytest.raises(Exception):
        sm.interview = "openai:gpt-4"  # type: ignore[misc]


def test_stage_models_fields_overridable() -> None:
    sm = StageModels(interview="openai:gpt-4o", codegen="gemini:gemini-2.0-flash")

    assert sm.interview == "openai:gpt-4o"
    assert sm.codegen == "gemini:gemini-2.0-flash"
    # unchanged defaults
    assert sm.performance == _SONNET_ID
