import pytest
from app.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        database_url="postgresql+asyncpg://test:test@localhost:5432/test",
        slack_bot_token="xoxb-test",
        slack_channel_id="C0TEST",
    )
