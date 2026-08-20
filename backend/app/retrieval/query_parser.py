import datetime as dt
import re
from dataclasses import dataclass
from enum import StrEnum

from app.jsonutil import extract_json_object
from app.providers.base import Generator

DATE_MODE_MAX_DAYS = 7
RECENT_DAYS_FOR_VAGUE = 3

CJK_DIGITS = {
    "零": 0, "一": 1, "兩": 2, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}  # fmt: skip

# 刻意不放單獨的「天」「月」「年」：天然氣、月營收、年增率都會誤觸發，
# 每次誤觸發都是一次多餘的 LLM 呼叫。
TEMPORAL_HINTS = re.compile(
    r"昨|今|前天|明天|週|周|星期|禮拜|"
    r"幾天|幾週|幾個月|個月|那天|當天|某天|月初|月底|年初|年底|去年|今年|明年|"
    r"最近|近期|過去|之前|以來|期間|何時|什麼時候|"
    r"month|week|day|year|yesterday|today|recent|lately|ago|since|during|when",
    re.IGNORECASE,
)

LLM_SYSTEM = (
    "你是查詢解析器。使用者會給你今天的日期與一個問題，請判斷問題想查詢的日期區間。\n"
    "只輸出一個 JSON 物件："
    '{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}；'
    '若問題沒有指涉任何時間範圍，輸出 {"start": null, "end": null}。'
)


class QueryMode(StrEnum):
    DATE = "date"
    HYBRID = "hybrid"
    SEMANTIC = "semantic"


@dataclass(frozen=True)
class DateRange:
    start: dt.date
    end: dt.date

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"end {self.end} is before start {self.start}")

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1


@dataclass(frozen=True)
class QueryPlan:
    question: str
    topic: str
    date_range: DateRange | None
    mode: QueryMode
    date_source: str


def parse_number(text: str) -> int | None:
    text = text.strip()
    if text.isdigit():
        return int(text)
    if not text or any(char not in CJK_DIGITS for char in text):
        return None
    if "十" not in text:
        return CJK_DIGITS[text] if len(text) == 1 else None

    index = text.index("十")
    tens = CJK_DIGITS[text[index - 1]] if index > 0 else 1
    ones = CJK_DIGITS[text[index + 1]] if index < len(text) - 1 else 0
    return tens * 10 + ones


def _month_day(month: int, day: int, today: dt.date) -> dt.date:
    try:
        candidate = dt.date(today.year, month, day)
    except ValueError as exc:
        raise ValueError(f"invalid month/day: {month}/{day}") from exc
    return candidate if candidate <= today else dt.date(today.year - 1, month, day)


def _week_start(day: dt.date) -> dt.date:
    return day - dt.timedelta(days=day.weekday())


def _month_start(day: dt.date) -> dt.date:
    return day.replace(day=1)


NUM = r"(\d+|[零一兩二三四五六七八九十]+)"

_RULES: list[tuple[re.Pattern[str], object]] = [
    (
        re.compile(r"(\d{4})-(\d{2})-(\d{2})\s*(?:到|至|~|～|—)\s*(\d{4})-(\d{2})-(\d{2})"),
        lambda m, today: DateRange(
            dt.date(int(m[1]), int(m[2]), int(m[3])), dt.date(int(m[4]), int(m[5]), int(m[6]))
        ),
    ),
    (
        re.compile(
            r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日號]?\s*(?:到|至|~|～|—)\s*(?:(\d{1,2})\s*月\s*)?(\d{1,2})\s*[日號]"
        ),
        lambda m, today: DateRange(
            _month_day(int(m[1]), int(m[2]), today),
            _month_day(int(m[3] or m[1]), int(m[4]), today),
        ),
    ),
    (
        re.compile(r"(\d{4})-(\d{2})-(\d{2})"),
        lambda m, today: _single(dt.date(int(m[1]), int(m[2]), int(m[3]))),
    ),
    (
        re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日號]"),
        lambda m, today: _single(_month_day(int(m[1]), int(m[2]), today)),
    ),
    (re.compile(r"大前天"), lambda m, today: _single(today - dt.timedelta(days=3))),
    (re.compile(r"前天"), lambda m, today: _single(today - dt.timedelta(days=2))),
    (
        re.compile(r"昨天|昨日|yesterday", re.IGNORECASE),
        lambda m, today: _single(today - dt.timedelta(days=1)),
    ),
    (re.compile(r"今天|今日|today", re.IGNORECASE), lambda m, today: _single(today)),
    (
        re.compile(rf"(?:最近|近|過去|這)\s*{NUM}\s*(天|日|週|周|星期|禮拜|個月)"),
        lambda m, today: _recent(m, today),
    ),
    (
        re.compile(r"(?:last|past|recent)\s+(\d+)\s+(day|week|month)s?", re.IGNORECASE),
        lambda m, today: _recent_en(m, today),
    ),
    (
        re.compile(r"上(?:個)?(?:週|周|星期|禮拜)|last\s+week", re.IGNORECASE),
        lambda m, today: _last_week(today),
    ),
    (
        re.compile(r"(?:本|這|這個)(?:週|周|星期|禮拜)|this\s+week", re.IGNORECASE),
        lambda m, today: DateRange(_week_start(today), today),
    ),
    (re.compile(r"上(?:個)?月|last\s+month", re.IGNORECASE), lambda m, today: _last_month(today)),
    (
        re.compile(r"(?:本|這個?)月|this\s+month", re.IGNORECASE),
        lambda m, today: DateRange(_month_start(today), today),
    ),
    (
        re.compile(r"最近幾天|這幾天|前幾天"),
        lambda m, today: _recent_days(today, RECENT_DAYS_FOR_VAGUE),
    ),
    (
        re.compile(r"最近|近期|lately|recently", re.IGNORECASE),
        lambda m, today: _recent_days(today, DATE_MODE_MAX_DAYS),
    ),
]


def _single(day: dt.date) -> DateRange:
    return DateRange(day, day)


def _recent_days(today: dt.date, days: int) -> DateRange:
    return DateRange(today - dt.timedelta(days=days - 1), today)


def _unit_days(unit: str) -> int:
    if unit in {"天", "日", "day"}:
        return 1
    if unit in {"週", "周", "星期", "禮拜", "week"}:
        return 7
    return 30


def _recent(match: re.Match[str], today: dt.date) -> DateRange:
    count = parse_number(match[1])
    if count is None or count < 1:
        raise ValueError(f"unparsable count: {match[1]!r}")
    return _recent_days(today, count * _unit_days(match[2]))


def _recent_en(match: re.Match[str], today: dt.date) -> DateRange:
    return _recent_days(today, int(match[1]) * _unit_days(match[2].lower()))


def _last_week(today: dt.date) -> DateRange:
    this_monday = _week_start(today)
    return DateRange(this_monday - dt.timedelta(days=7), this_monday - dt.timedelta(days=1))


def _last_month(today: dt.date) -> DateRange:
    last_day = _month_start(today) - dt.timedelta(days=1)
    return DateRange(_month_start(last_day), last_day)


def parse_date_expression(
    question: str, today: dt.date
) -> tuple[DateRange, tuple[int, int]] | None:
    for pattern, build in _RULES:
        match = pattern.search(question)
        if match is None:
            continue
        try:
            return build(match, today), match.span()
        except ValueError:
            continue
    return None


def _stitch(prefix: str, suffix: str) -> str:
    if not prefix or not suffix:
        return prefix or suffix
    separator = " " if (prefix[-1].isascii() and suffix[0].isascii()) else ""
    return prefix + separator + suffix


def extract_topic(question: str, span: tuple[int, int] | None) -> str:
    """移除日期片語後留下的主題，用於向量檢索。"""
    if span is None:
        return question.strip()
    prefix = re.sub(r"[的之在於，,、\s]+$", "", question[: span[0]])
    suffix = re.sub(r"^[的之，,、\s]+", "", question[span[1] :])
    return _stitch(prefix, suffix).strip() or question.strip()


def choose_mode(date_range: DateRange | None) -> QueryMode:
    if date_range is None:
        return QueryMode.SEMANTIC
    return QueryMode.DATE if date_range.days <= DATE_MODE_MAX_DAYS else QueryMode.HYBRID


def looks_temporal(question: str) -> bool:
    return TEMPORAL_HINTS.search(question) is not None


def parse_llm_range(raw: str) -> DateRange | None:
    payload = extract_json_object(raw)
    if not payload:
        return None
    try:
        start = dt.date.fromisoformat(payload["start"])
        end = dt.date.fromisoformat(payload["end"])
    except (KeyError, TypeError, ValueError):
        return None
    return DateRange(start, end) if start <= end else None


async def build_plan(
    question: str, today: dt.date, generator: Generator | None = None
) -> QueryPlan:
    matched = parse_date_expression(question, today)
    if matched is not None:
        date_range, span = matched
        return QueryPlan(
            question, extract_topic(question, span), date_range, choose_mode(date_range), "rule"
        )

    if generator is not None and looks_temporal(question):
        prompt = f"今天是 {today.isoformat()}。\n\n問題：{question}"
        try:
            date_range = parse_llm_range(await generator.generate(LLM_SYSTEM, prompt))
        except Exception:
            date_range = None
        if date_range is not None:
            return QueryPlan(question, question.strip(), date_range, choose_mode(date_range), "llm")

    return QueryPlan(question, question.strip(), None, QueryMode.SEMANTIC, "none")
