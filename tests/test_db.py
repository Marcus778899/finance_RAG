import pytest
from app.config import Settings
from app.db import create_engine, create_session_factory, session_scope
from app.models import EMBEDDING_DIM
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


def test_embedding_dim_matches_settings_default():
    assert Settings.model_fields["embed_dim"].default == EMBEDDING_DIM


def test_create_engine_uses_settings_url(settings):
    engine = create_engine(settings)
    assert engine.url.database == "finance_rag_test"
    assert engine.url.drivername == "postgresql+asyncpg"


async def test_session_scope_yields_usable_session(db_url):
    engine = create_async_engine(db_url)
    factory = create_session_factory(engine)
    try:
        async with session_scope(factory) as session:
            assert (await session.execute(text("SELECT 1"))).scalar_one() == 1
    finally:
        await engine.dispose()


async def test_session_scope_rolls_back_on_error(db_url):
    engine = create_async_engine(db_url)
    factory = create_session_factory(engine)
    try:
        with pytest.raises(RuntimeError):
            async with session_scope(factory) as session:
                await session.execute(text("SELECT 1"))
                raise RuntimeError("boom")
    finally:
        await engine.dispose()
