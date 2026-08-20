"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-08-19
"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIM = 1024


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "slack_messages",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("channel_id", sa.String(length=32), nullable=False),
        sa.Column("slack_ts", sa.String(length=32), nullable=False),
        sa.Column("author", sa.String(length=128), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("permalink", sa.Text(), nullable=True),
        sa.Column("posted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("channel_id", "slack_ts", name="uq_slack_messages_channel_ts"),
    )
    op.create_index("ix_slack_messages_posted_at", "slack_messages", ["posted_at"])

    op.create_table(
        "daily_digests",
        sa.Column("digest_date", sa.Date(), nullable=False),
        sa.Column("summary_zh", sa.Text(), nullable=False),
        sa.Column("key_topics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "generated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("digest_date"),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column(
            "source",
            sa.Enum("message", "digest", name="chunk_source"),
            nullable=False,
        ),
        sa.Column("message_id", sa.BigInteger(), nullable=True),
        sa.Column("digest_date", sa.Date(), nullable=True),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("posted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.ForeignKeyConstraint(["message_id"], ["slack_messages.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["digest_date"], ["daily_digests.digest_date"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("message_id", "seq", name="uq_chunks_message_seq"),
        sa.UniqueConstraint("digest_date", "seq", name="uq_chunks_digest_seq"),
        sa.CheckConstraint(
            "(source = 'message' AND message_id IS NOT NULL AND digest_date IS NULL)"
            " OR (source = 'digest' AND digest_date IS NOT NULL AND message_id IS NULL)",
            name="ck_chunks_source_matches_parent",
        ),
    )
    op.create_index("ix_chunks_posted_at", "chunks", ["posted_at"])


def downgrade() -> None:
    op.drop_index("ix_chunks_posted_at", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("daily_digests")
    op.drop_index("ix_slack_messages_posted_at", table_name="slack_messages")
    op.drop_table("slack_messages")
    sa.Enum(name="chunk_source").drop(op.get_bind())
