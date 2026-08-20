import datetime as dt

import pytest

from app.providers import FakeGenerator
from app.retrieval.query_parser import (
    DATE_MODE_MAX_DAYS,
    DateRange,
    QueryMode,
    build_plan,
    choose_mode,
    extract_topic,
    looks_temporal,
    parse_date_expression,
    parse_llm_range,
    parse_number,
)

TODAY = dt.date(2026, 8, 20)


def parsed(question: str, today: dt.date = TODAY) -> DateRange | None:
    result = parse_date_expression(question, today)
    return result[0] if result else None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("3", 3),
        ("15", 15),
        ("三", 3),
        ("兩", 2),
        ("十", 10),
        ("十五", 15),
        ("二十", 20),
        ("三十五", 35),
    ],
)
def test_parse_number(text, expected):
    assert parse_number(text) == expected


@pytest.mark.parametrize("text", ["", "abc", "三四五", "壹"])
def test_parse_number_rejects_unparsable(text):
    assert parse_number(text) is None


@pytest.mark.parametrize(
    "question, start, end",
    [
        ("昨天的美國財經新聞", "2026-08-19", "2026-08-19"),
        ("今天台股如何", "2026-08-20", "2026-08-20"),
        ("前天的Fed決議", "2026-08-18", "2026-08-18"),
        ("大前天呢", "2026-08-17", "2026-08-17"),
        ("這週Fed立場", "2026-08-17", "2026-08-20"),
        ("上週美股走勢", "2026-08-10", "2026-08-16"),
        ("本月台積電", "2026-08-01", "2026-08-20"),
        ("上個月通膨", "2026-07-01", "2026-07-31"),
        ("最近三天重點", "2026-08-18", "2026-08-20"),
        ("最近5天", "2026-08-16", "2026-08-20"),
        ("過去兩週半導體", "2026-08-07", "2026-08-20"),
        ("最近一個月原油", "2026-07-22", "2026-08-20"),
        ("8月15日發生什麼", "2026-08-15", "2026-08-15"),
        ("8月15號的新聞", "2026-08-15", "2026-08-15"),
        ("2026-08-18的新聞", "2026-08-18", "2026-08-18"),
        ("8月15日到8月18日的美股", "2026-08-15", "2026-08-18"),
        ("2026-08-15 到 2026-08-18", "2026-08-15", "2026-08-18"),
        ("最近幾天", "2026-08-18", "2026-08-20"),
        ("what happened yesterday", "2026-08-19", "2026-08-19"),
        ("last week summary", "2026-08-10", "2026-08-16"),
        ("last 3 days", "2026-08-18", "2026-08-20"),
        ("past 2 weeks", "2026-08-07", "2026-08-20"),
    ],
)
def test_rule_layer_parses_common_phrasings(question, start, end):
    assert parsed(question) == DateRange(dt.date.fromisoformat(start), dt.date.fromisoformat(end))


def test_no_date_expression_returns_none():
    assert parsed("關於半導體有提過什麼") is None


def test_month_day_without_year_rolls_back_when_in_the_future():
    assert parsed("12月25日的新聞", today=dt.date(2026, 1, 5)) == DateRange(
        dt.date(2025, 12, 25), dt.date(2025, 12, 25)
    )


def test_impossible_month_day_falls_through_instead_of_crashing():
    assert parsed("2月30日的新聞") is None


def test_last_month_across_year_boundary():
    assert parsed("上個月", today=dt.date(2026, 1, 15)) == DateRange(
        dt.date(2025, 12, 1), dt.date(2025, 12, 31)
    )


def test_this_week_on_a_monday_is_a_single_day():
    monday = dt.date(2026, 8, 17)
    assert parsed("這週", today=monday) == DateRange(monday, monday)


def test_specific_phrasings_win_over_generic_recent():
    assert parsed("最近三天") == DateRange(dt.date(2026, 8, 18), TODAY)
    assert parsed("最近有什麼") == DateRange(dt.date(2026, 8, 14), TODAY)


def test_qianthian_is_not_swallowed_by_yesterday_rule():
    assert parsed("前天") == DateRange(dt.date(2026, 8, 18), dt.date(2026, 8, 18))


def test_date_range_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="before start"):
        DateRange(dt.date(2026, 8, 19), dt.date(2026, 8, 17))


def test_date_range_day_count_is_inclusive():
    assert DateRange(dt.date(2026, 8, 17), dt.date(2026, 8, 19)).days == 3


def test_extract_topic_removes_the_date_phrase():
    span = parse_date_expression("請問昨天的美國有什麼重大財經新聞", TODAY)[1]
    assert extract_topic("請問昨天的美國有什麼重大財經新聞", span) == "請問美國有什麼重大財經新聞"


def test_extract_topic_keeps_english_spacing():
    span = parse_date_expression("what happened yesterday", TODAY)[1]
    assert extract_topic("what happened yesterday", span) == "what happened"


def test_extract_topic_falls_back_when_nothing_is_left():
    span = parse_date_expression("最近幾天", TODAY)[1]
    assert extract_topic("最近幾天", span) == "最近幾天"


def test_extract_topic_without_a_span_returns_the_question():
    assert extract_topic("  半導體  ", None) == "半導體"


@pytest.mark.parametrize(
    "days, expected",
    [
        (1, QueryMode.DATE),
        (DATE_MODE_MAX_DAYS, QueryMode.DATE),
        (DATE_MODE_MAX_DAYS + 1, QueryMode.HYBRID),
    ],
)
def test_mode_is_chosen_by_span_length(days, expected):
    start = TODAY - dt.timedelta(days=days - 1)
    assert choose_mode(DateRange(start, TODAY)) is expected


def test_mode_without_a_date_is_semantic():
    assert choose_mode(None) is QueryMode.SEMANTIC


@pytest.mark.parametrize("question", ["昨天如何", "這個月呢", "what about last year", "最近怎樣"])
def test_looks_temporal_detects_time_words(question):
    assert looks_temporal(question)


@pytest.mark.parametrize("question", ["半導體產業", "台積電營收", "Nvidia earnings"])
def test_looks_temporal_ignores_plain_topics(question):
    assert not looks_temporal(question)


def test_parse_llm_range_accepts_valid_json():
    assert parse_llm_range('{"start": "2026-08-01", "end": "2026-08-05"}') == DateRange(
        dt.date(2026, 8, 1), dt.date(2026, 8, 5)
    )


@pytest.mark.parametrize(
    "raw",
    [
        '{"start": null, "end": null}',
        '{"start": "2026-08-05", "end": "2026-08-01"}',
        '{"start": "not-a-date", "end": "2026-08-01"}',
        '{"end": "2026-08-01"}',
        "沒有時間範圍",
        "[]",
    ],
)
def test_parse_llm_range_rejects_bad_payloads(raw):
    assert parse_llm_range(raw) is None


async def test_build_plan_prefers_the_rule_layer_and_skips_the_llm():
    generator = FakeGenerator(['{"start": "1999-01-01", "end": "1999-01-01"}'])
    plan = await build_plan("昨天的美國財經新聞", TODAY, generator)

    assert plan.date_source == "rule"
    assert plan.date_range == DateRange(dt.date(2026, 8, 19), dt.date(2026, 8, 19))
    assert generator.prompts == []


async def test_build_plan_falls_back_to_the_llm_for_unusual_phrasing():
    generator = FakeGenerator(['{"start": "2026-08-01", "end": "2026-08-03"}'])
    plan = await build_plan("中元節前後那幾天的行情", TODAY, generator)

    assert plan.date_source == "llm"
    assert plan.date_range == DateRange(dt.date(2026, 8, 1), dt.date(2026, 8, 3))


async def test_build_plan_does_not_call_the_llm_for_plain_topics():
    generator = FakeGenerator(['{"start": "2026-08-01", "end": "2026-08-03"}'])
    plan = await build_plan("關於半導體有提過什麼", TODAY, generator)

    assert plan.mode is QueryMode.SEMANTIC
    assert plan.date_source == "none"
    assert generator.prompts == []


async def test_build_plan_degrades_to_semantic_when_the_llm_is_unusable():
    plan = await build_plan("中元節前後那幾天的行情", TODAY, FakeGenerator(["抱歉我不知道"]))
    assert plan.mode is QueryMode.SEMANTIC
    assert plan.date_source == "none"


async def test_build_plan_degrades_to_semantic_when_the_llm_raises():
    class Broken(FakeGenerator):
        async def stream(self, system, prompt):
            raise RuntimeError("ollama down")
            yield ""

    plan = await build_plan("中元節前後那幾天的行情", TODAY, Broken())
    assert plan.mode is QueryMode.SEMANTIC


async def test_build_plan_without_a_generator_stays_on_rules():
    plan = await build_plan("中元節前後那幾天的行情", TODAY)
    assert plan.mode is QueryMode.SEMANTIC
    assert plan.date_source == "none"


def test_zero_count_falls_through_to_the_generic_recent_rule():
    assert parsed("最近零天的新聞") == DateRange(dt.date(2026, 8, 14), TODAY)


def test_unparsable_count_falls_through_to_the_generic_recent_rule():
    assert parsed("最近三四五天的新聞") == DateRange(dt.date(2026, 8, 14), TODAY)
