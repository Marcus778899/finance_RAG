from __future__ import annotations

import datetime as dt
import enum

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIM = 1024


class Base(DeclarativeBase):
    pass


class ChunkSource(enum.StrEnum):
    MESSAGE = "message"
    DIGEST = "digest"


class SlackMessage(Base):
    __tablename__ = "slack_messages"
    __table_args__ = (
        UniqueConstraint("channel_id", "slack_ts", name="uq_slack_messages_channel_ts"),
        Index("ix_slack_messages_posted_at", "posted_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    channel_id: Mapped[str] = mapped_column(String(32))
    slack_ts: Mapped[str] = mapped_column(String(32))
    author: Mapped[str | None] = mapped_column(String(128))
    text: Mapped[str] = mapped_column(Text)
    permalink: Mapped[str | None] = mapped_column(Text)
    posted_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    ingested_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="message", cascade="all, delete-orphan"
    )


class DailyDigest(Base):
    __tablename__ = "daily_digests"

    digest_date: Mapped[dt.date] = mapped_column(Date, primary_key=True)
    summary_zh: Mapped[str] = mapped_column(Text)
    key_topics: Mapped[list[str]] = mapped_column(JSONB, default=list)
    generated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="digest", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        CheckConstraint(
            "(source = 'message' AND message_id IS NOT NULL AND digest_date IS NULL)"
            " OR (source = 'digest' AND digest_date IS NOT NULL AND message_id IS NULL)",
            name="ck_chunks_source_matches_parent",
        ),
        UniqueConstraint("message_id", "seq", name="uq_chunks_message_seq"),
        UniqueConstraint("digest_date", "seq", name="uq_chunks_digest_seq"),
        Index("ix_chunks_posted_at", "posted_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source: Mapped[ChunkSource] = mapped_column(
        Enum(ChunkSource, name="chunk_source", values_callable=lambda e: [m.value for m in e])
    )
    message_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("slack_messages.id", ondelete="CASCADE")
    )
    digest_date: Mapped[dt.date | None] = mapped_column(
        Date, ForeignKey("daily_digests.digest_date", ondelete="CASCADE")
    )
    seq: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    posted_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))

    message: Mapped[SlackMessage | None] = relationship(back_populates="chunks")
    digest: Mapped[DailyDigest | None] = relationship(back_populates="chunks")
