import pytest
from pydantic import ValidationError

from cuda_engine.services.llm.capabilities import ProviderCapabilities


def test_conservative_defaults() -> None:
    caps = ProviderCapabilities(provider="test")
    assert caps.prompt_caching is False
    assert caps.tool_use is False
    assert caps.max_context == 200_000


def test_explicit_fields_round_trip() -> None:
    caps = ProviderCapabilities(
        provider="anthropic", prompt_caching=True, tool_use=True, max_context=100_000
    )
    assert caps.provider == "anthropic"
    assert caps.prompt_caching is True
    assert caps.tool_use is True
    assert caps.max_context == 100_000


def test_frozen_raises_on_mutation() -> None:
    caps = ProviderCapabilities(provider="test")
    with pytest.raises(ValidationError):
        caps.provider = "other"  # type: ignore[misc]
