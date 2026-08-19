import pytest
from app.config import Settings

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
