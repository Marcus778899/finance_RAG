from app.providers.claude import MAX_TOKENS, ClaudeGenerator


class FakeStream:
    def __init__(self, texts):
        self._texts = texts

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    @property
    def text_stream(self):
        async def iterator():
            for text in self._texts:
                yield text

        return iterator()


class FakeMessages:
    def __init__(self, texts):
        self._texts = texts
        self.kwargs: dict = {}

    def stream(self, **kwargs):
        self.kwargs = kwargs
        return FakeStream(self._texts)


class FakeAnthropic:
    def __init__(self, texts):
        self.messages = FakeMessages(texts)


async def test_streams_text_deltas():
    client = FakeAnthropic(["美股", "收紅"])
    generator = ClaudeGenerator(model="claude-opus-5", client=client)
    assert [chunk async for chunk in generator.stream("sys", "問題")] == ["美股", "收紅"]


async def test_request_uses_adaptive_thinking_and_configured_model():
    client = FakeAnthropic(["ok"])
    generator = ClaudeGenerator(model="claude-opus-5", client=client)
    await generator.generate("sys", "問題")

    kwargs = client.messages.kwargs
    assert kwargs["model"] == "claude-opus-5"
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["system"] == "sys"
    assert kwargs["max_tokens"] == MAX_TOKENS
    assert kwargs["messages"] == [{"role": "user", "content": "問題"}]
