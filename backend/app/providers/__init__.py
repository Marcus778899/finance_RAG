import httpx

from app.config import Settings
from app.providers.base import (
    ContextOverflowError,
    Embedder,
    EmbeddingDimensionError,
    Generator,
    ProviderError,
)
from app.providers.fake import FakeEmbedder, FakeGenerator
from app.providers.ollama import OllamaEmbedder, OllamaGenerator, create_client

__all__ = [
    "ContextOverflowError",
    "Embedder",
    "EmbeddingDimensionError",
    "FakeEmbedder",
    "FakeGenerator",
    "Generator",
    "OllamaEmbedder",
    "OllamaGenerator",
    "ProviderError",
    "create_client",
    "create_embedder",
    "create_generator",
]


def create_embedder(settings: Settings, client: httpx.AsyncClient | None = None) -> Embedder:
    return OllamaEmbedder(
        client or create_client(settings),
        model=settings.embed_model,
        dimension=settings.embed_dim,
    )


def create_generator(settings: Settings, client: httpx.AsyncClient | None = None) -> Generator:
    backend = settings.generator_backend
    if backend == "ollama":
        return OllamaGenerator(
            client or create_client(settings),
            model=settings.generator_model,
            num_ctx=settings.generator_num_ctx,
        )
    if backend == "claude":
        from app.providers.claude import ClaudeGenerator

        return ClaudeGenerator(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key.get_secret_value(),
        )
    raise ValueError(f"unknown generator backend: {backend!r}")
