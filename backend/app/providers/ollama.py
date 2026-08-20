import json
from collections.abc import AsyncIterator, Sequence

import httpx

from app.config import Settings
from app.providers.base import (
    ContextOverflowError,
    EmbeddingDimensionError,
    Generator,
    ProviderError,
)
from app.tokens import estimate_tokens

CONTEXT_SAFETY_MARGIN = 512


class OllamaEmbedder:
    def __init__(self, client: httpx.AsyncClient, model: str, dimension: int) -> None:
        self._client = client
        self._model = model
        self.dimension = dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        response = await self._client.post(
            "/api/embed", json={"model": self._model, "input": list(texts)}
        )
        response.raise_for_status()
        embeddings = response.json().get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise ProviderError(f"unexpected /api/embed payload for {len(texts)} inputs")

        for vector in embeddings:
            if len(vector) != self.dimension:
                raise EmbeddingDimensionError(self.dimension, len(vector))
        return embeddings


class OllamaGenerator(Generator):
    def __init__(self, client: httpx.AsyncClient, model: str, num_ctx: int) -> None:
        self._client = client
        self._model = model
        self._num_ctx = num_ctx

    def _payload(self, system: str, prompt: str) -> dict:
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            # 不要送 think: false。qwen3 在 Ollama 0.32 下仍會思考，而該旗標會關掉
            # template 的 thinking 解析，讓原始思考文字掉進 content。省略它，
            # 思考會被分到 message.thinking，我們只讀 message.content 即可。
            "options": {"num_ctx": self._num_ctx},
        }

    def _guard_context(self, system: str, prompt: str) -> None:
        estimated = estimate_tokens(system) + estimate_tokens(prompt) + CONTEXT_SAFETY_MARGIN
        if estimated > self._num_ctx:
            raise ContextOverflowError(estimated, self._num_ctx)

    async def stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        self._guard_context(system, prompt)

        async with self._client.stream(
            "POST", "/api/chat", json=self._payload(system, prompt)
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ProviderError(f"malformed /api/chat line: {line!r}") from exc
                if error := event.get("error"):
                    raise ProviderError(f"ollama returned an error: {error}")
                if content := event.get("message", {}).get("content"):
                    yield content
                if event.get("done"):
                    break


def create_client(settings: Settings) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=httpx.Timeout(300.0))
