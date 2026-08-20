import datetime as dt
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError, IntegrityError

from app.models import EMBEDDING_DIM, Chunk, ChunkSource, DailyDigest, SlackMessage

TAIPEI = ZoneInfo("Asia/Taipei")


def embedding(seed: float = 0.1) -> list[float]:
    return [seed] * EMBEDDING_DIM


def make_message(ts: str = "1755561600.000100", **kwargs) -> SlackMessage:
    defaults = {
        "channel_id": "C0TEST",
        "slack_ts": ts,
        "author": "finance-bot",
        "text": "Fed 維持利率不變",
        "permalink": "https://slack.com/archives/C0TEST/p1755561600000100",
        "posted_at": dt.datetime(2026, 8, 19, 8, 0, tzinfo=TAIPEI),
    }
    return SlackMessage(**defaults | kwargs)


async def test_vector_extension_is_installed(db_session):
    result = await db_session.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'"))
    assert result.scalar_one() == 1


async def test_message_roundtrip_preserves_timezone(db_session):
    db_session.add(make_message())
    await db_session.flush()

    stored = (await db_session.execute(select(SlackMessage))).scalar_one()
    assert stored.posted_at.astimezone(TAIPEI) == dt.datetime(2026, 8, 19, 8, 0, tzinfo=TAIPEI)
    assert stored.ingested_at is not None


async def test_duplicate_slack_ts_in_same_channel_is_rejected(db_session):
    db_session.add(make_message())
    await db_session.flush()

    db_session.add(make_message(text="重複"))
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def test_same_ts_in_different_channel_is_allowed(db_session):
    db_session.add(make_message())
    db_session.add(make_message(channel_id="C0OTHER"))
    await db_session.flush()

    assert len((await db_session.execute(select(SlackMessage))).scalars().all()) == 2


async def test_message_chunk_roundtrip(db_session):
    message = make_message()
    db_session.add(message)
    await db_session.flush()

    db_session.add(
        Chunk(
            source=ChunkSource.MESSAGE,
            message_id=message.id,
            seq=0,
            content="Fed 維持利率不變",
            token_count=12,
            posted_at=message.posted_at,
            embedding=embedding(),
        )
    )
    await db_session.flush()

    stored = (await db_session.execute(select(Chunk))).scalar_one()
    assert stored.source is ChunkSource.MESSAGE
    assert len(stored.embedding) == EMBEDDING_DIM


async def test_digest_chunk_roundtrip(db_session):
    digest = DailyDigest(
        digest_date=dt.date(2026, 8, 19),
        summary_zh="Fed 按兵不動，美股收紅。",
        key_topics=["Fed", "美股"],
    )
    db_session.add(digest)
    await db_session.flush()

    db_session.add(
        Chunk(
            source=ChunkSource.DIGEST,
            digest_date=digest.digest_date,
            seq=0,
            content=digest.summary_zh,
            token_count=20,
            posted_at=dt.datetime(2026, 8, 19, 23, 59, tzinfo=TAIPEI),
            embedding=embedding(0.2),
        )
    )
    await db_session.flush()

    stored = (await db_session.execute(select(Chunk))).scalar_one()
    assert stored.source is ChunkSource.DIGEST
    assert stored.message_id is None


async def test_chunk_source_must_match_parent(db_session):
    message = make_message()
    db_session.add(message)
    await db_session.flush()

    db_session.add(
        Chunk(
            source=ChunkSource.DIGEST,
            message_id=message.id,
            seq=0,
            content="來源與父層不一致",
            token_count=5,
            posted_at=message.posted_at,
            embedding=embedding(),
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def test_chunk_without_any_parent_is_rejected(db_session):
    db_session.add(
        Chunk(
            source=ChunkSource.MESSAGE,
            seq=0,
            content="孤兒 chunk",
            token_count=5,
            posted_at=dt.datetime(2026, 8, 19, 8, 0, tzinfo=TAIPEI),
            embedding=embedding(),
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def test_duplicate_seq_within_message_is_rejected(db_session):
    message = make_message()
    db_session.add(message)
    await db_session.flush()

    for _ in range(2):
        db_session.add(
            Chunk(
                source=ChunkSource.MESSAGE,
                message_id=message.id,
                seq=0,
                content="重複 seq",
                token_count=5,
                posted_at=message.posted_at,
                embedding=embedding(),
            )
        )
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def test_deleting_message_cascades_to_chunks(db_session):
    message = make_message()
    db_session.add(message)
    await db_session.flush()
    db_session.add(
        Chunk(
            source=ChunkSource.MESSAGE,
            message_id=message.id,
            seq=0,
            content="會被連帶刪除",
            token_count=5,
            posted_at=message.posted_at,
            embedding=embedding(),
        )
    )
    await db_session.flush()

    await db_session.delete(message)
    await db_session.flush()

    assert (await db_session.execute(select(Chunk))).scalars().all() == []


async def test_wrong_embedding_dimension_is_rejected(db_session):
    message = make_message()
    db_session.add(message)
    await db_session.flush()

    db_session.add(
        Chunk(
            source=ChunkSource.MESSAGE,
            message_id=message.id,
            seq=0,
            content="維度錯誤",
            token_count=5,
            posted_at=message.posted_at,
            embedding=[0.1] * 512,
        )
    )
    with pytest.raises(DBAPIError):
        await db_session.flush()


async def test_cosine_distance_ordering(db_session):
    message = make_message()
    db_session.add(message)
    await db_session.flush()

    near = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
    far = [0.0] * (EMBEDDING_DIM - 1) + [1.0]
    for seq, vector in enumerate((near, far)):
        db_session.add(
            Chunk(
                source=ChunkSource.MESSAGE,
                message_id=message.id,
                seq=seq,
                content=f"chunk-{seq}",
                token_count=5,
                posted_at=message.posted_at,
                embedding=vector,
            )
        )
    await db_session.flush()

    ordered = (
        (
            await db_session.execute(
                select(Chunk).order_by(Chunk.embedding.cosine_distance(near)).limit(1)
            )
        )
        .scalars()
        .all()
    )
    assert ordered[0].content == "chunk-0"
