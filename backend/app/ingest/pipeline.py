import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingest.chunker import chunk_text
from app.ingest.digest import generate_digest
from app.ingest.slack import SlackMessageData, SlackReader
from app.models import Chunk, ChunkSource, DailyDigest, SlackMessage
from app.providers.base import Embedder, Generator
from app.timeutil import day_bounds


@dataclass(frozen=True)
class IngestResult:
    day: dt.date
    messages: int
    chunks: int
    digest_topics: list[str]

    @property
    def is_empty(self) -> bool:
        return self.messages == 0


def days_to_ingest(
    today: dt.date, *, day: dt.date | None = None, backfill: int | None = None
) -> list[dt.date]:
    """未指定時預設抓昨天，因為當天的訊息通常還沒進完。"""
    if day is not None and backfill is not None:
        raise ValueError("day and backfill are mutually exclusive")
    if day is not None:
        return [day]
    if backfill is not None:
        if backfill < 1:
            raise ValueError("backfill must be at least 1")
        return [today - dt.timedelta(days=offset) for offset in range(backfill, 0, -1)]
    return [today - dt.timedelta(days=1)]


async def _upsert_messages(
    session: AsyncSession, messages: list[SlackMessageData]
) -> dict[str, int]:
    statement = pg_insert(SlackMessage).values(
        [
            {
                "channel_id": message.channel_id,
                "slack_ts": message.slack_ts,
                "author": message.author,
                "text": message.text,
                "permalink": message.permalink,
                "posted_at": message.posted_at,
            }
            for message in messages
        ]
    )
    statement = statement.on_conflict_do_update(
        index_elements=[SlackMessage.channel_id, SlackMessage.slack_ts],
        set_={
            "author": statement.excluded.author,
            "text": statement.excluded.text,
            "permalink": statement.excluded.permalink,
            "posted_at": statement.excluded.posted_at,
        },
    ).returning(SlackMessage.id, SlackMessage.slack_ts)

    rows = (await session.execute(statement)).all()
    return {slack_ts: message_id for message_id, slack_ts in rows}


async def _replace_message_chunks(
    session: AsyncSession,
    embedder: Embedder,
    messages: list[SlackMessageData],
    ids_by_ts: dict[str, int],
) -> int:
    pending: list[tuple[int, int, str, int, dt.datetime]] = []
    for message in messages:
        for seq, chunk in enumerate(chunk_text(message.text)):
            pending.append(
                (
                    ids_by_ts[message.slack_ts],
                    seq,
                    chunk.content,
                    chunk.token_count,
                    message.posted_at,
                )
            )

    await session.execute(delete(Chunk).where(Chunk.message_id.in_(list(ids_by_ts.values()))))
    vectors = await embedder.embed([content for _, _, content, _, _ in pending])
    session.add_all(
        [
            Chunk(
                source=ChunkSource.MESSAGE,
                message_id=message_id,
                seq=seq,
                content=content,
                token_count=token_count,
                posted_at=posted_at,
                embedding=vector,
            )
            for (message_id, seq, content, token_count, posted_at), vector in zip(
                pending, vectors, strict=True
            )
        ]
    )
    return len(pending)


async def _replace_digest(
    session: AsyncSession,
    embedder: Embedder,
    generator: Generator,
    day: dt.date,
    messages: list[SlackMessageData],
    end: dt.datetime,
) -> list[str]:
    digest = await generate_digest(generator, day, [message.text for message in messages])

    statement = pg_insert(DailyDigest).values(
        digest_date=day, summary_zh=digest.summary_zh, key_topics=digest.key_topics
    )
    statement = statement.on_conflict_do_update(
        index_elements=[DailyDigest.digest_date],
        set_={
            "summary_zh": statement.excluded.summary_zh,
            "key_topics": statement.excluded.key_topics,
        },
    )
    await session.execute(statement)

    await session.execute(delete(Chunk).where(Chunk.digest_date == day))
    chunks = chunk_text(digest.summary_zh)
    if chunks:
        vectors = await embedder.embed([chunk.content for chunk in chunks])
        session.add_all(
            [
                Chunk(
                    source=ChunkSource.DIGEST,
                    digest_date=day,
                    seq=seq,
                    content=chunk.content,
                    token_count=chunk.token_count,
                    posted_at=end - dt.timedelta(seconds=1),
                    embedding=vector,
                )
                for seq, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
            ]
        )
    return digest.key_topics


async def ingest_day(
    session: AsyncSession,
    reader: SlackReader,
    embedder: Embedder,
    generator: Generator,
    *,
    channel_id: str,
    day: dt.date,
    tz: ZoneInfo,
) -> IngestResult:
    start, end = day_bounds(day, tz)
    messages = await reader.fetch_range(channel_id, start, end)
    if not messages:
        return IngestResult(day=day, messages=0, chunks=0, digest_topics=[])

    ids_by_ts = await _upsert_messages(session, messages)
    chunk_count = await _replace_message_chunks(session, embedder, messages, ids_by_ts)
    topics = await _replace_digest(session, embedder, generator, day, messages, end)
    await session.flush()

    return IngestResult(day=day, messages=len(messages), chunks=chunk_count, digest_topics=topics)


async def ingest_days(
    session: AsyncSession,
    reader: SlackReader,
    embedder: Embedder,
    generator: Generator,
    *,
    channel_id: str,
    days: list[dt.date],
    tz: ZoneInfo,
) -> list[IngestResult]:
    return [
        await ingest_day(
            session,
            reader,
            embedder,
            generator,
            channel_id=channel_id,
            day=day,
            tz=tz,
        )
        for day in days
    ]


async def stored_days(session: AsyncSession) -> list[dt.date]:
    rows = await session.execute(select(DailyDigest.digest_date).order_by(DailyDigest.digest_date))
    return list(rows.scalars().all())
