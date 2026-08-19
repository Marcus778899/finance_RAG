from zoneinfo import ZoneInfo

import pytest
from app.config import Settings, get_settings
from pydantic import ValidationError


def test_tzinfo_resolves_configured_timezone(make_settings):
    assert make_settings(timezone="Asia/Taipei").tzinfo == ZoneInfo("Asia/Taipei")


def test_database_url_is_built_from_components(settings):
    assert settings.database_url == (
        "postgresql+asyncpg://test_user:test_password@localhost:5432/finance_rag_test"
    )


def test_database_url_escapes_special_characters(make_settings):
    settings = make_settings(postgres_password="p@ss w/ord")
    assert "p%40ss+w%2Ford" in settings.database_url


def test_password_is_not_exposed_in_repr(settings):
    assert "test_password" not in repr(settings)


def test_password_is_required():
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_get_settings_is_cached(env):
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()
