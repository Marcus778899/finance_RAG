import hashlib
import math
from collections.abc import AsyncIterator, Sequence

from app.providers.base import Generator


class FakeEmbedder:
    def __init__(self, dimension: int = 1024) -> None:
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def vector_for(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        raw = [
            (digest[i % len(digest)] ^ (i * 31 % 251)) / 255.0 - 0.5 for i in range(self.dimension)
        ]
        norm = math.sqrt(sum(value * value for value in raw)) or 1.0
        return [value / norm for value in raw]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.vector_for(text) for text in texts]


class FakeGenerator(Generator):
    def __init__(self, chunks: Sequence[str] = ("這是", "測試", "回答")) -> None:
        self._chunks = list(chunks)
        self.prompts: list[tuple[str, str]] = []

    async def stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        self.prompts.append((system, prompt))
        for chunk in self._chunks:
            yield chunk
