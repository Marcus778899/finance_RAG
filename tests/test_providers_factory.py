import pytest

from app.providers import create_embedder, create_generator
from app.providers.claude import ClaudeGenerator
from app.providers.ollama import OllamaEmbedder, OllamaGenerator


def test_create_embedder_uses_configured_model_and_dimension(settings):
    embedder = create_embedder(settings)
    assert isinstance(embedder, OllamaEmbedder)
    assert embedder.dimension == settings.embed_dim


def test_create_generator_defaults_to_ollama(settings):
    generator = create_generator(settings)
    assert isinstance(generator, OllamaGenerator)
    assert generator._num_ctx == settings.generator_num_ctx


def test_create_generator_can_switch_to_claude(make_settings):
    settings = make_settings(generator_backend="claude", anthropic_api_key="sk-test")
    assert isinstance(create_generator(settings), ClaudeGenerator)


def test_unknown_backend_is_rejected(make_settings):
    with pytest.raises(ValueError, match="unknown generator backend"):
        create_generator(make_settings(generator_backend="llamafile"))
