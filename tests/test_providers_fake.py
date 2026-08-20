from app.providers import FakeEmbedder, FakeGenerator


async def test_fake_embedder_is_deterministic():
    embedder = FakeEmbedder()
    first = await embedder.embed(["聯準會"])
    second = await embedder.embed(["聯準會"])
    assert first == second


async def test_fake_embedder_differs_per_text():
    embedder = FakeEmbedder()
    vectors = await embedder.embed(["聯準會", "台積電"])
    assert vectors[0] != vectors[1]


async def test_fake_embedder_returns_unit_vectors_of_configured_dimension():
    embedder = FakeEmbedder(dimension=8)
    (vector,) = await embedder.embed(["x"])
    assert len(vector) == 8
    assert abs(sum(value * value for value in vector) - 1.0) < 1e-9


async def test_fake_embedder_records_calls():
    embedder = FakeEmbedder()
    await embedder.embed(["a", "b"])
    assert embedder.calls == [["a", "b"]]


async def test_fake_generator_streams_and_records_prompts():
    generator = FakeGenerator(["A", "B"])
    chunks = [chunk async for chunk in generator.stream("sys", "使用者問題")]
    assert chunks == ["A", "B"]
    assert generator.prompts == [("sys", "使用者問題")]


async def test_generate_joins_the_stream():
    assert await FakeGenerator(["A", "B"]).generate("sys", "q") == "AB"
