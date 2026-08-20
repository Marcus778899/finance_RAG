import json

import httpx
import pytest

from app.providers.base import (
    ContextOverflowError,
    EmbeddingDimensionError,
    ProviderError,
)
from app.providers.ollama import OllamaEmbedder, OllamaGenerator

DIM = 1024


def ndjson(*events: dict) -> bytes:
    return "\n".join(json.dumps(event) for event in events).encode()


def make_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ollama")


def chat_events(*contents: str) -> bytes:
    events = [{"message": {"content": content}, "done": False} for content in contents]
    events.append({"message": {"content": ""}, "done": True})
    return ndjson(*events)


async def test_embed_returns_vectors():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == "bge-m3"
        assert payload["input"] == ["聯準會", "台積電"]
        return httpx.Response(200, json={"embeddings": [[0.1] * DIM, [0.2] * DIM]})

    embedder = OllamaEmbedder(make_client(handler), model="bge-m3", dimension=DIM)
    vectors = await embedder.embed(["聯準會", "台積電"])
    assert [len(vector) for vector in vectors] == [DIM, DIM]


async def test_embed_short_circuits_on_empty_input():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not call ollama")

    embedder = OllamaEmbedder(make_client(handler), model="bge-m3", dimension=DIM)
    assert await embedder.embed([]) == []


async def test_embed_rejects_wrong_dimension():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[0.1] * 512]})

    embedder = OllamaEmbedder(make_client(handler), model="bge-m3", dimension=DIM)
    with pytest.raises(EmbeddingDimensionError) as excinfo:
        await embedder.embed(["x"])
    assert excinfo.value.expected == DIM
    assert excinfo.value.actual == 512


async def test_embed_rejects_count_mismatch():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[0.1] * DIM]})

    embedder = OllamaEmbedder(make_client(handler), model="bge-m3", dimension=DIM)
    with pytest.raises(ProviderError):
        await embedder.embed(["a", "b"])


async def test_embed_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    embedder = OllamaEmbedder(make_client(handler), model="bge-m3", dimension=DIM)
    with pytest.raises(httpx.HTTPStatusError):
        await embedder.embed(["x"])


async def test_generator_always_sends_num_ctx_and_omits_think_flag():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=chat_events("答案"))

    generator = OllamaGenerator(make_client(handler), model="qwen3:4b", num_ctx=32768)
    await generator.generate("system", "問題")

    assert seen["options"]["num_ctx"] == 32768
    assert "think" not in seen
    assert seen["stream"] is True
    assert seen["messages"][0]["role"] == "system"


async def test_generator_streams_content_chunks():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=chat_events("美股", "收紅"))

    generator = OllamaGenerator(make_client(handler), model="qwen3:4b", num_ctx=32768)
    chunks = [chunk async for chunk in generator.stream("system", "問題")]
    assert chunks == ["美股", "收紅"]


async def test_generator_refuses_prompt_that_exceeds_num_ctx():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not reach ollama")

    generator = OllamaGenerator(make_client(handler), model="qwen3:4b", num_ctx=4096)
    with pytest.raises(ContextOverflowError) as excinfo:
        await generator.generate("system", "美股收紅。" * 2000)
    assert excinfo.value.num_ctx == 4096
    assert excinfo.value.estimated > 4096


async def test_generator_accepts_prompt_within_num_ctx():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=chat_events("ok"))

    generator = OllamaGenerator(make_client(handler), model="qwen3:4b", num_ctx=32768)
    assert await generator.generate("system", "美股收紅。" * 2000) == "ok"


async def test_generator_surfaces_ollama_error_events():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=ndjson({"error": "model not found"}))

    generator = OllamaGenerator(make_client(handler), model="qwen3:4b", num_ctx=32768)
    with pytest.raises(ProviderError, match="model not found"):
        await generator.generate("system", "問題")


async def test_generator_rejects_malformed_lines():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json\n")

    generator = OllamaGenerator(make_client(handler), model="qwen3:4b", num_ctx=32768)
    with pytest.raises(ProviderError, match="malformed"):
        await generator.generate("system", "問題")


async def test_generator_ignores_blank_lines():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\n" + chat_events("ok") + b"\n\n")

    generator = OllamaGenerator(make_client(handler), model="qwen3:4b", num_ctx=32768)
    assert await generator.generate("system", "問題") == "ok"


async def test_generator_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"")

    generator = OllamaGenerator(make_client(handler), model="qwen3:4b", num_ctx=32768)
    with pytest.raises(httpx.HTTPStatusError):
        await generator.generate("system", "問題")


async def test_generator_ignores_thinking_deltas():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=ndjson(
                {"message": {"role": "assistant", "thinking": "先想一下"}, "done": False},
                {"message": {"role": "assistant", "thinking": "再想一下"}, "done": False},
                {"message": {"role": "assistant", "content": "美股收紅"}, "done": False},
                {"message": {"content": ""}, "done": True},
            ),
        )

    generator = OllamaGenerator(make_client(handler), model="qwen3:4b", num_ctx=32768)
    assert await generator.generate("system", "問題") == "美股收紅"
