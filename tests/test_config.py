from zoneinfo import ZoneInfo

from app.config import Settings, get_settings


def test_tzinfo_resolves_configured_timezone():
    assert Settings(timezone="Asia/Taipei").tzinfo == ZoneInfo("Asia/Taipei")


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
