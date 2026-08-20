from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable


class ProviderError(RuntimeError):
    pass


class EmbeddingDimensionError(ProviderError):
    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(f"embedding dimension mismatch: expected {expected}, got {actual}")
        self.expected = expected
        self.actual = actual


class ContextOverflowError(ProviderError):
    def __init__(self, estimated: int, num_ctx: int) -> None:
        super().__init__(
            f"prompt is about {estimated} tokens but num_ctx is {num_ctx}; "
            "Ollama would silently drop the overflow"
        )
        self.estimated = estimated
        self.num_ctx = num_ctx


@runtime_checkable
class Embedder(Protocol):
    dimension: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class Generator(ABC):
    @abstractmethod
    def stream(self, system: str, prompt: str) -> AsyncIterator[str]: ...

    async def generate(self, system: str, prompt: str) -> str:
        return "".join([chunk async for chunk in self.stream(system, prompt)])
