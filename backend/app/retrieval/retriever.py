import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, ChunkSource, SlackMessage
from app.providers.base import Embedder
from app.retrieval.budget import fit_within_budget
from app.retrieval.query_parser import DateRange, QueryMode, QueryPlan
from app.timeutil import local_date, range_bounds

DEFAULT_TOP_K = 12


@dataclass(frozen=True)
class RetrievedChunk:
    content: str
    day: dt.date
    posted_at: dt.datetime
    source: ChunkSource
    token_count: int
    author: str | None = None
    permalink: str | None = None


@dataclass(frozen=True)
class RetrievalResult:
    plan: QueryPlan
    chunks: list[RetrievedChunk]
    dropped_days: list[dt.date]
    truncated_day: dt.date | None

    @property
    def degraded(self) -> bool:
        return bool(self.dropped_days) or self.truncated_day is not None

    @property
    def days_covered(self) -> list[dt.date]:
        return sorted({chunk.day for chunk in self.chunks})


def _base_query() -> Select:
    return select(
        Chunk.content,
        Chunk.posted_at,
        Chunk.source,
        Chunk.token_count,
        SlackMessage.author,
        SlackMessage.permalink,
    ).outerjoin(SlackMessage, Chunk.message_id == SlackMessage.id)


def _rows_to_chunks(rows: Sequence, tz: ZoneInfo) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            content=content,
            day=local_date(posted_at, tz),
            posted_at=posted_at,
            source=source,
            token_count=token_count,
            author=author,
            permalink=permalink,
        )
        for content, posted_at, source, token_count, author, permalink in rows
    ]


async def _fetch_window(
    session: AsyncSession,
    date_range: DateRange,
    tz: ZoneInfo,
    sources: tuple[ChunkSource, ...],
) -> list[RetrievedChunk]:
    start, end = range_bounds(date_range.start, date_range.end, tz)
    query = (
        _base_query()
        .where(Chunk.posted_at >= start, Chunk.posted_at < end, Chunk.source.in_(sources))
        .order_by(Chunk.posted_at, Chunk.seq)
    )
    return _rows_to_chunks((await session.execute(query)).all(), tz)


async def _vector_search(
    session: AsyncSession,
    embedding: list[float],
    tz: ZoneInfo,
    *,
    top_k: int,
    date_range: DateRange | None = None,
    sources: tuple[ChunkSource, ...] = (ChunkSource.MESSAGE, ChunkSource.DIGEST),
) -> list[RetrievedChunk]:
    query = _base_query().where(Chunk.source.in_(sources))
    if date_range is not None:
        start, end = range_bounds(date_range.start, date_range.end, tz)
        query = query.where(Chunk.posted_at >= start, Chunk.posted_at < end)
    query = query.order_by(Chunk.embedding.cosine_distance(embedding)).limit(top_k)
    return _rows_to_chunks((await session.execute(query)).all(), tz)


def _deduplicate(chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: set[tuple[dt.datetime, str]] = set()
    unique = []
    for chunk in chunks:
        key = (chunk.posted_at, chunk.content)
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique


async def retrieve(
    session: AsyncSession,
    plan: QueryPlan,
    embedder: Embedder,
    *,
    tz: ZoneInfo,
    budget: int,
    top_k: int = DEFAULT_TOP_K,
) -> RetrievalResult:
    if plan.mode is QueryMode.DATE:
        candidates = await _fetch_window(session, plan.date_range, tz, (ChunkSource.MESSAGE,))
    elif plan.mode is QueryMode.HYBRID:
        (embedding,) = await embedder.embed([plan.topic])
        digests = await _fetch_window(session, plan.date_range, tz, (ChunkSource.DIGEST,))
        details = await _vector_search(
            session,
            embedding,
            tz,
            top_k=top_k,
            date_range=plan.date_range,
            sources=(ChunkSource.MESSAGE,),
        )
        candidates = _deduplicate([*digests, *details])
    else:
        (embedding,) = await embedder.embed([plan.topic])
        candidates = await _vector_search(session, embedding, tz, top_k=top_k)

    outcome = fit_within_budget(sorted(candidates, key=lambda chunk: chunk.posted_at), budget)
    return RetrievalResult(
        plan=plan,
        chunks=outcome.kept,
        dropped_days=outcome.dropped_days,
        truncated_day=outcome.truncated_day,
    )
