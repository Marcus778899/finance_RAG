import datetime as dt
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import func, select

from app.ingest.pipeline import days_to_ingest, ingest_day, ingest_days, stored_days
from app.ingest.slack import SlackMessageData
from app.models import Chunk, ChunkSource, DailyDigest, SlackMessage
from app.providers import FakeEmbedder, FakeGenerator
from app.timeutil import UTC

TAIPEI = ZoneInfo("Asia/Taipei")
DAY = dt.date(2026, 8, 19)
CHANNEL = "C0TEST"
DIGEST_JSON = '{"summary": "美股收紅，聯準會按兵不動。", "topics": ["美股", "Fed"]}'


class FakeReader:
    def __init__(self, by_day: dict[dt.date, list[SlackMessageData]] | None = None) -> None:
        self.by_day = by_day or {}
        self.calls: list[tuple[dt.datetime, dt.datetime]] = []

    async def fetch_range(self, channel_id, start, end):
        self.calls.append((start, end))
        return list(self.by_day.get(start.astimezone(TAIPEI).date(), []))


def slack_message(hour: int = 9, text: str = "美股收紅。台積電上漲。") -> SlackMessageData:
    posted_at = dt.datetime(2026, 8, 19, hour, 0, tzinfo=TAIPEI).astimezone(UTC)
    return SlackMessageData(
        channel_id=CHANNEL,
        slack_ts=f"{posted_at.timestamp():.6f}",
        author="finance-bot",
        text=text,
        permalink="https://slack/p1",
        posted_at=posted_at,
    )


async def run_ingest(session, messages, day=DAY, generator=None):
    return await ingest_day(
        session,
        FakeReader({day: messages}),
        FakeEmbedder(),
        generator or FakeGenerator([DIGEST_JSON]),
        channel_id=CHANNEL,
        day=day,
        tz=TAIPEI,
    )


async def count(session, model) -> int:
    return (await session.execute(select(func.count()).select_from(model))).scalar_one()


def test_days_to_ingest_defaults_to_yesterday():
    assert days_to_ingest(DAY) == [dt.date(2026, 8, 18)]


def test_days_to_ingest_with_explicit_date():
    assert days_to_ingest(DAY, day=dt.date(2026, 1, 1)) == [dt.date(2026, 1, 1)]


def test_days_to_ingest_backfill_is_chronological_and_excludes_today():
    assert days_to_ingest(DAY, backfill=3) == [
        dt.date(2026, 8, 16),
        dt.date(2026, 8, 17),
        dt.date(2026, 8, 18),
    ]


def test_days_to_ingest_rejects_conflicting_options():
    with pytest.raises(ValueError, match="mutually exclusive"):
        days_to_ingest(DAY, day=DAY, backfill=2)


def test_days_to_ingest_rejects_non_positive_backfill():
    with pytest.raises(ValueError, match="at least 1"):
        days_to_ingest(DAY, backfill=0)


async def test_empty_day_stores_nothing(db_session):
    result = await run_ingest(db_session, [])

    assert result.is_empty
    assert await count(db_session, SlackMessage) == 0
    assert await count(db_session, DailyDigest) == 0


async def test_ingest_stores_messages_chunks_and_digest(db_session):
    result = await run_ingest(db_session, [slack_message()])

    assert result.messages == 1
    assert result.chunks >= 1
    assert result.digest_topics == ["美股", "Fed"]
    assert await count(db_session, SlackMessage) == 1
    assert await count(db_session, DailyDigest) == 1


async def test_digest_produces_its_own_chunks(db_session):
    await run_ingest(db_session, [slack_message()])

    digest_chunks = (
        (await db_session.execute(select(Chunk).where(Chunk.source == ChunkSource.DIGEST)))
        .scalars()
        .all()
    )
    assert digest_chunks
    assert all(chunk.digest_date == DAY for chunk in digest_chunks)
    assert "美股收紅" in digest_chunks[0].content


async def test_chunks_carry_the_message_posted_at(db_session):
    message = slack_message()
    await run_ingest(db_session, [message])

    chunks = (
        (await db_session.execute(select(Chunk).where(Chunk.source == ChunkSource.MESSAGE)))
        .scalars()
        .all()
    )
    assert all(chunk.posted_at == message.posted_at for chunk in chunks)


async def test_reader_is_queried_with_taipei_day_bounds(db_session):
    reader = FakeReader({DAY: []})
    await ingest_day(
        db_session,
        reader,
        FakeEmbedder(),
        FakeGenerator([DIGEST_JSON]),
        channel_id=CHANNEL,
        day=DAY,
        tz=TAIPEI,
    )
    start, end = reader.calls[0]
    assert start == dt.datetime(2026, 8, 18, 16, 0, tzinfo=UTC)
    assert end == dt.datetime(2026, 8, 19, 16, 0, tzinfo=UTC)


async def test_rerunning_the_same_day_is_idempotent(db_session):
    message = slack_message()
    first = await run_ingest(db_session, [message])
    before = (await count(db_session, SlackMessage), await count(db_session, Chunk))

    second = await run_ingest(db_session, [message])
    after = (await count(db_session, SlackMessage), await count(db_session, Chunk))

    assert before == after
    assert first.chunks == second.chunks


async def test_rerun_updates_edited_message_text_and_chunks(db_session):
    message = slack_message(text="舊內容。")
    await run_ingest(db_session, [message])

    edited = SlackMessageData(**{**message.__dict__, "text": "新內容，已更正。"})
    await run_ingest(db_session, [edited])

    stored = (await db_session.execute(select(SlackMessage))).scalar_one()
    assert stored.text == "新內容，已更正。"

    contents = (
        (await db_session.execute(select(Chunk.content).where(Chunk.source == ChunkSource.MESSAGE)))
        .scalars()
        .all()
    )
    assert any("新內容" in content for content in contents)
    assert not any("舊內容" in content for content in contents)


async def test_rerun_replaces_the_digest(db_session):
    await run_ingest(db_session, [slack_message()])
    await run_ingest(
        db_session,
        [slack_message()],
        generator=FakeGenerator(['{"summary": "改寫後的摘要。", "topics": ["新主題"]}']),
    )

    digest = (await db_session.execute(select(DailyDigest))).scalar_one()
    assert digest.summary_zh == "改寫後的摘要。"
    assert digest.key_topics == ["新主題"]


async def test_every_chunk_gets_an_embedding_of_the_right_size(db_session):
    await run_ingest(db_session, [slack_message()])

    chunks = (await db_session.execute(select(Chunk))).scalars().all()
    assert chunks
    assert all(len(chunk.embedding) == 1024 for chunk in chunks)


async def test_embedder_is_called_in_batches_not_per_chunk(db_session):
    embedder = FakeEmbedder()
    await ingest_day(
        db_session,
        FakeReader({DAY: [slack_message(), slack_message(hour=10)]}),
        embedder,
        FakeGenerator([DIGEST_JSON]),
        channel_id=CHANNEL,
        day=DAY,
        tz=TAIPEI,
    )
    assert len(embedder.calls) == 2
    assert len(embedder.calls[0]) >= 2


async def test_digest_failure_leaves_nothing_committed(db_session):
    class Failing(FakeGenerator):
        async def stream(self, system, prompt):
            raise RuntimeError("ollama down")
            yield ""

    with pytest.raises(RuntimeError, match="ollama down"):
        await run_ingest(db_session, [slack_message()], generator=Failing())

    await db_session.rollback()
    assert await count(db_session, SlackMessage) == 0


async def test_ingest_days_processes_each_day(db_session):
    other = dt.date(2026, 8, 18)
    reader = FakeReader({DAY: [slack_message()], other: []})
    results = await ingest_days(
        db_session,
        reader,
        FakeEmbedder(),
        FakeGenerator([DIGEST_JSON, DIGEST_JSON]),
        channel_id=CHANNEL,
        days=[other, DAY],
        tz=TAIPEI,
    )
    assert [r.day for r in results] == [other, DAY]
    assert [r.is_empty for r in results] == [True, False]


async def test_stored_days_lists_digest_dates_in_order(db_session):
    assert await stored_days(db_session) == []
    await run_ingest(db_session, [slack_message()])
    assert await stored_days(db_session) == [DAY]
