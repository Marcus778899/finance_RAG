import datetime as dt
from zoneinfo import ZoneInfo

import pytest

from app.timeutil import (
    UTC,
    day_bounds,
    from_slack_ts,
    local_date,
    range_bounds,
    to_slack_ts,
)

TAIPEI = ZoneInfo("Asia/Taipei")
NEW_YORK = ZoneInfo("America/New_York")


def test_day_bounds_shifts_by_offset():
    start, end = day_bounds(dt.date(2026, 8, 19), TAIPEI)
    assert start == dt.datetime(2026, 8, 18, 16, 0, tzinfo=UTC)
    assert end == dt.datetime(2026, 8, 19, 16, 0, tzinfo=UTC)


def test_day_bounds_span_is_exactly_one_day():
    start, end = day_bounds(dt.date(2026, 8, 19), TAIPEI)
    assert end - start == dt.timedelta(days=1)


def test_day_bounds_across_dst_transition_is_not_24h():
    start, end = day_bounds(dt.date(2026, 3, 8), NEW_YORK)
    assert end - start == dt.timedelta(hours=23)


def test_range_bounds_is_inclusive_of_end_day():
    start, end = range_bounds(dt.date(2026, 8, 17), dt.date(2026, 8, 19), TAIPEI)
    assert start == dt.datetime(2026, 8, 16, 16, 0, tzinfo=UTC)
    assert end == dt.datetime(2026, 8, 19, 16, 0, tzinfo=UTC)


def test_range_bounds_rejects_inverted_range():
    with pytest.raises(ValueError, match="before start"):
        range_bounds(dt.date(2026, 8, 19), dt.date(2026, 8, 17), TAIPEI)


def test_range_bounds_across_year_boundary():
    start, end = range_bounds(dt.date(2025, 12, 31), dt.date(2026, 1, 1), TAIPEI)
    assert end - start == dt.timedelta(days=2)


def test_local_date_uses_target_timezone():
    just_after_midnight_taipei = dt.datetime(2026, 8, 18, 16, 30, tzinfo=UTC)
    assert local_date(just_after_midnight_taipei, TAIPEI) == dt.date(2026, 8, 19)
    assert local_date(just_after_midnight_taipei, NEW_YORK) == dt.date(2026, 8, 18)


def test_local_date_rejects_naive_datetime():
    with pytest.raises(ValueError, match="naive"):
        local_date(dt.datetime(2026, 8, 19, 8, 0), TAIPEI)


def test_slack_ts_roundtrip():
    moment = from_slack_ts("1755561600.000100")
    assert moment.tzinfo is UTC
    assert to_slack_ts(moment) == "1755561600.000100"


def test_slack_ts_maps_to_expected_taipei_date():
    assert local_date(from_slack_ts("1755561600.000100"), TAIPEI) == dt.date(2025, 8, 19)
