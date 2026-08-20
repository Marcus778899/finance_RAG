import datetime as dt
from zoneinfo import ZoneInfo

UTC = dt.UTC


def day_bounds(day: dt.date, tz: ZoneInfo) -> tuple[dt.datetime, dt.datetime]:
    """回傳該地方日在 UTC 的 [起, 迄) 邊界。"""
    start = dt.datetime.combine(day, dt.time.min, tzinfo=tz)
    return start.astimezone(UTC), (start + dt.timedelta(days=1)).astimezone(UTC)


def range_bounds(start: dt.date, end: dt.date, tz: ZoneInfo) -> tuple[dt.datetime, dt.datetime]:
    """回傳含頭含尾的地方日區間在 UTC 的 [起, 迄) 邊界。"""
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    return day_bounds(start, tz)[0], day_bounds(end, tz)[1]


def local_date(moment: dt.datetime, tz: ZoneInfo) -> dt.date:
    if moment.tzinfo is None:
        raise ValueError("naive datetime has no unambiguous local date")
    return moment.astimezone(tz).date()


def from_slack_ts(slack_ts: str) -> dt.datetime:
    return dt.datetime.fromtimestamp(float(slack_ts), tz=UTC)


def to_slack_ts(moment: dt.datetime) -> str:
    return f"{moment.timestamp():.6f}"
