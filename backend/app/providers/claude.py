from collections.abc import AsyncIterator
from typing import Any

from app.providers.base import Generator

MAX_TOKENS = 32000


def default_client(api_key: str) -> Any:
    from anthropic import AsyncAnthropic

    return AsyncAnthropic(api_key=api_key) if api_key else AsyncAnthropic()


class ClaudeGenerator(Generator):
    def __init__(self, model: str, api_key: str = "", client: Any | None = None) -> None:
        self._model = model
        self._client = client if client is not None else default_client(api_key)

    async def stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=MAX_TOKENS,
            system=system,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text
