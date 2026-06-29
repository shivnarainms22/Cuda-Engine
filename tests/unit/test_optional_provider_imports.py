"""Importing the optional provider adapter modules must never raise,
even when the underlying SDKs (openai, google-genai) are absent.

The adapters use lazy imports: the SDK is only imported inside __init__
when no client is injected — so the module-level import is always safe.
"""
import importlib


def test_openai_client_module_imports_without_error() -> None:
    mod = importlib.import_module("cuda_engine.services.llm.openai_client")
    assert hasattr(mod, "OpenAIClient")


def test_gemini_client_module_imports_without_error() -> None:
    mod = importlib.import_module("cuda_engine.services.llm.gemini_client")
    assert hasattr(mod, "GeminiClient")


def test_openai_compatible_module_imports_without_error() -> None:
    mod = importlib.import_module("cuda_engine.services.llm.openai_compatible")
    assert hasattr(mod, "OpenAICompatibleClient")
