import datetime as dt
from zoneinfo import ZoneInfo

import pytest

from app.models import Chunk, ChunkSource, DailyDigest, SlackMessage
from app.providers import FakeEmbedder
from app.retrieval.query_parser import DateRange, QueryMode, QueryPlan
from app.retrieval.retriever import retrieve
from app.timeutil import UTC
from app.tokens import estimate_tokens

TAIPEI = ZoneInfo("Asia/Taipei")
BUDGET = 100_000


def plan_for(mode: QueryMode, start=None, end=None, topic="美股") -> QueryPlan:
    date_range = DateRange(start, end or start) if start else None
    return QueryPlan("問題", topic, date_range, mode, "rule")


async def seed(session, embedder, day, texts, *, digest=None, hour=9):
    posted = dt.datetime(day.year, day.month, day.day, hour, tzinfo=TAIPEI).astimezone(UTC)
    message = SlackMessage(
        channel_id="C0TEST",
        slack_ts=f"{posted.timestamp():.6f}",
        author="finance-bot",
        text="\n\n".join(texts),
        permalink=f"https://slack/{day}",
        posted_at=posted,
    )
    session.add(message)
    await session.flush()

    for seq, (text, vector) in enumerate(zip(texts, await embedder.embed(texts), strict=True)):
        session.add(
            Chunk(
                source=ChunkSource.MESSAGE,
                message_id=message.id,
                seq=seq,
                content=text,
                token_count=estimate_tokens(text),
                posted_at=posted,
                embedding=vector,
            )
        )

    if digest:
        session.add(DailyDigest(digest_date=day, summary_zh=digest, key_topics=[]))
        await session.flush()
        (vector,) = await embedder.embed([digest])
        session.add(
            Chunk(
                source=ChunkSource.DIGEST,
                digest_date=day,
                seq=0,
                content=digest,
                token_count=estimate_tokens(digest),
                posted_at=posted + dt.timedelta(hours=1),
                embedding=vector,
            )
        )
    await session.flush()


@pytest.fixture
def embedder():
    return FakeEmbedder()


async def run(session, embedder, plan, budget=BUDGET, top_k=12):
    return await retrieve(session, plan, embedder, tz=TAIPEI, budget=budget, top_k=top_k)


async def test_date_mode_returns_only_that_day(db_session, embedder):
    await seed(db_session, embedder, dt.date(2026, 8, 18), ["前天新聞"])
    await seed(db_session, embedder, dt.date(2026, 8, 19), ["昨天新聞A", "昨天新聞B"])
    await seed(db_session, embedder, dt.date(2026, 8, 20), ["今天新聞"])

    result = await run(db_session, embedder, plan_for(QueryMode.DATE, dt.date(2026, 8, 19)))
    assert [chunk.content for chunk in result.chunks] == ["昨天新聞A", "昨天新聞B"]
    assert result.days_covered == [dt.date(2026, 8, 19)]


async def test_date_mode_excludes_digest_chunks(db_session, embedder):
    await seed(db_session, embedder, dt.date(2026, 8, 19), ["昨天新聞"], digest="昨天摘要")

    result = await run(db_session, embedder, plan_for(QueryMode.DATE, dt.date(2026, 8, 19)))
    assert [chunk.source for chunk in result.chunks] == [ChunkSource.MESSAGE]


async def test_date_mode_does_not_embed_anything(db_session, embedder):
    await seed(db_session, embedder, dt.date(2026, 8, 19), ["昨天新聞"])
    embedder.calls.clear()

    await run(db_session, embedder, plan_for(QueryMode.DATE, dt.date(2026, 8, 19)))
    assert embedder.calls == []


async def test_date_boundary_follows_taipei_not_utc(db_session, embedder):
    await seed(db_session, embedder, dt.date(2026, 8, 19), ["台北時間凌晨零點半"], hour=0)

    taipei_day = await run(db_session, embedder, plan_for(QueryMode.DATE, dt.date(2026, 8, 19)))
    previous_day = await run(db_session, embedder, plan_for(QueryMode.DATE, dt.date(2026, 8, 18)))

    assert len(taipei_day.chunks) == 1
    assert previous_day.chunks == []


async def test_date_mode_spanning_several_days_is_chronological(db_session, embedder):
    for day, text in ((17, "十七"), (18, "十八"), (19, "十九")):
        await seed(db_session, embedder, dt.date(2026, 8, day), [text])

    result = await run(
        db_session, embedder, plan_for(QueryMode.DATE, dt.date(2026, 8, 17), dt.date(2026, 8, 19))
    )
    assert [chunk.content for chunk in result.chunks] == ["十七", "十八", "十九"]


async def test_empty_range_returns_nothing_without_degrading(db_session, embedder):
    result = await run(db_session, embedder, plan_for(QueryMode.DATE, dt.date(2026, 8, 19)))
    assert result.chunks == []
    assert not result.degraded


async def test_hybrid_mode_includes_digests_and_vector_hits(db_session, embedder):
    for day in range(1, 11):
        await seed(
            db_session,
            embedder,
            dt.date(2026, 8, day),
            [f"第{day}天的原文內容"],
            digest=f"第{day}天摘要",
        )

    plan = plan_for(QueryMode.HYBRID, dt.date(2026, 8, 1), dt.date(2026, 8, 10), topic="第3天摘要")
    result = await run(db_session, embedder, plan)

    sources = {chunk.source for chunk in result.chunks}
    assert ChunkSource.DIGEST in sources
    assert ChunkSource.MESSAGE in sources
    assert embedder.calls[-1] == ["第3天摘要"]


async def test_hybrid_mode_stays_inside_the_date_range(db_session, embedder):
    await seed(db_session, embedder, dt.date(2026, 7, 1), ["範圍外內容"], digest="範圍外摘要")
    for day in range(1, 11):
        await seed(db_session, embedder, dt.date(2026, 8, day), [f"第{day}天"], digest=f"摘要{day}")

    plan = plan_for(QueryMode.HYBRID, dt.date(2026, 8, 1), dt.date(2026, 8, 10))
    result = await run(db_session, embedder, plan)

    assert all(chunk.day.month == 8 for chunk in result.chunks)


async def test_semantic_mode_finds_the_closest_chunk_anywhere(db_session, embedder):
    await seed(db_session, embedder, dt.date(2026, 8, 1), ["半導體產業展望"])
    await seed(db_session, embedder, dt.date(2026, 8, 19), ["原油價格下跌"])

    plan = plan_for(QueryMode.SEMANTIC, topic="半導體產業展望")
    result = await run(db_session, embedder, plan, top_k=1)

    assert [chunk.content for chunk in result.chunks] == ["半導體產業展望"]


async def test_semantic_mode_respects_top_k(db_session, embedder):
    await seed(db_session, embedder, dt.date(2026, 8, 19), [f"新聞{i}" for i in range(10)])

    result = await run(db_session, embedder, plan_for(QueryMode.SEMANTIC), top_k=3)
    assert len(result.chunks) == 3


async def test_chunks_carry_author_and_permalink(db_session, embedder):
    await seed(db_session, embedder, dt.date(2026, 8, 19), ["昨天新聞"])

    result = await run(db_session, embedder, plan_for(QueryMode.DATE, dt.date(2026, 8, 19)))
    assert result.chunks[0].author == "finance-bot"
    assert result.chunks[0].permalink == "https://slack/2026-08-19"


async def test_digest_chunks_have_no_permalink(db_session, embedder):
    await seed(db_session, embedder, dt.date(2026, 8, 19), ["原文"], digest="摘要")

    plan = plan_for(QueryMode.HYBRID, dt.date(2026, 8, 19), dt.date(2026, 8, 19))
    result = await run(db_session, embedder, plan)
    digests = [chunk for chunk in result.chunks if chunk.source is ChunkSource.DIGEST]
    assert digests and digests[0].permalink is None


async def test_budget_drops_the_oldest_days_and_reports_it(db_session, embedder):
    for day in (17, 18, 19):
        await seed(db_session, embedder, dt.date(2026, 8, day), ["新聞內容" * 20])

    plan = plan_for(QueryMode.DATE, dt.date(2026, 8, 17), dt.date(2026, 8, 19))
    result = await run(db_session, embedder, plan, budget=120)

    assert result.degraded
    assert result.dropped_days == [dt.date(2026, 8, 17), dt.date(2026, 8, 18)]
    assert result.days_covered == [dt.date(2026, 8, 19)]


async def test_hybrid_deduplicates_overlapping_results(db_session, embedder):
    await seed(db_session, embedder, dt.date(2026, 8, 19), ["同一則內容"], digest="同一則內容")

    plan = plan_for(QueryMode.HYBRID, dt.date(2026, 8, 19), dt.date(2026, 8, 19))
    result = await run(db_session, embedder, plan)
    contents = [chunk.content for chunk in result.chunks]
    assert len(contents) == len(
        set(zip(contents, [c.posted_at for c in result.chunks], strict=True))
    )
