import asyncio
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from app.config import Settings
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from testcontainers.community.postgres import PostgresContainer

from alembic import command
from alembic.config import Config

ROOT = Path(__file__).resolve().parents[1]

ENV = {
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "finance_rag_test",
    "POSTGRES_USER": "test_user",
    "POSTGRES_PASSWORD": "test_password",
    "SLACK_BOT_TOKEN": "xoxb-test",
    "SLACK_CHANNEL_ID": "C0TEST",
}


@pytest.fixture
def make_settings():
    def factory(**overrides) -> Settings:
        return Settings(_env_file=None, **{k.lower(): v for k, v in ENV.items()} | overrides)

    return factory


@pytest.fixture
def settings(make_settings) -> Settings:
    return make_settings()


@pytest.fixture
def env(monkeypatch):
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)


def alembic_config(db_url: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "alembic"))
    config.attributes["db_url"] = db_url
    return config


def prepare_schema(db_url: str) -> None:
    config = alembic_config(db_url)
    command.upgrade(config, "head")
    command.downgrade(config, "base")
    command.upgrade(config, "head")


@pytest.fixture(scope="session")
def db_url() -> Iterator[str]:
    with PostgresContainer("pgvector/pgvector:pg17", driver="asyncpg") as postgres:
        url = postgres.get_connection_url()
        asyncio.run(_wait_for_db(url))
        prepare_schema(url)
        yield url


async def _wait_for_db(url: str) -> None:
    from sqlalchemy import text

    engine = create_async_engine(url)
    try:
        for attempt in range(30):
            try:
                async with engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
                return
            except Exception:
                if attempt == 29:
                    raise
                await asyncio.sleep(1)
    finally:
        await engine.dispose()


@pytest.fixture
async def db_session(db_url: str) -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(db_url)
    async with engine.connect() as connection:
        transaction = await connection.begin()
        session = AsyncSession(bind=connection, expire_on_commit=False)
        try:
            yield session
        finally:
            await session.close()
            if transaction.is_active:
                await transaction.rollback()
    await engine.dispose()
