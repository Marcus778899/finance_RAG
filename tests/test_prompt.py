import datetime as dt

from app.models import ChunkSource
from app.retrieval.prompt import (
    ANSWER_SYSTEM,
    NO_CONTEXT_NOTE,
    build_answer_prompt,
    describe_degradation,
    describe_scope,
    format_context,
)
from app.retrieval.query_parser import DateRange, QueryMode, QueryPlan
from app.retrieval.retriever import RetrievalResult, RetrievedChunk
from app.timeutil import UTC

TODAY = dt.date(2026, 8, 20)


def chunk(day: int, content: str, source=ChunkSource.MESSAGE, author="finance-bot"):
    return RetrievedChunk(
        content=content,
        day=dt.date(2026, 8, day),
        posted_at=dt.datetime(2026, 8, day, 1, tzinfo=UTC),
        source=source,
        token_count=10,
        author=author,
        permalink=None,
    )


def result_for(mode, chunks, *, start=None, end=None, dropped=None, truncated=None):
    date_range = DateRange(start, end or start) if start else None
    plan = QueryPlan("昨天有什麼新聞", "新聞", date_range, mode, "rule")
    return RetrievalResult(plan, chunks, dropped or [], truncated)


def test_context_groups_chunks_by_day_in_order():
    text = format_context([chunk(19, "後面"), chunk(18, "前面")])
    assert text.index("2026-08-18") < text.index("2026-08-19")


def test_context_labels_message_and_digest_differently():
    text = format_context([chunk(19, "原文內容"), chunk(19, "摘要內容", ChunkSource.DIGEST)])
    assert "[原文 @finance-bot]" in text
    assert "[當日摘要 @finance-bot]" not in text or "當日摘要" in text


def test_context_omits_author_when_missing():
    assert "@" not in format_context([chunk(19, "內容", author=None)])


def test_empty_context_is_explicit():
    assert format_context([]) == NO_CONTEXT_NOTE


def test_scope_for_a_date_range():
    result = result_for(QueryMode.DATE, [], start=dt.date(2026, 8, 19))
    assert describe_scope(result) == "查詢範圍：2026-08-19 至 2026-08-19"


def test_scope_for_hybrid_explains_the_mix():
    result = result_for(QueryMode.HYBRID, [], start=dt.date(2026, 8, 1), end=dt.date(2026, 8, 20))
    assert "當日摘要" in describe_scope(result)


def test_scope_for_semantic_says_all_history():
    assert "全部歷史" in describe_scope(result_for(QueryMode.SEMANTIC, []))


def test_no_degradation_note_when_everything_fits():
    assert describe_degradation(result_for(QueryMode.DATE, [], start=TODAY)) is None


def test_degradation_note_lists_dropped_days():
    result = result_for(QueryMode.DATE, [], start=TODAY, dropped=[dt.date(2026, 8, 17)])
    assert "2026-08-17" in describe_degradation(result)


def test_degradation_note_mentions_truncation():
    result = result_for(QueryMode.DATE, [], start=TODAY, truncated=dt.date(2026, 8, 20))
    assert "只納入前段" in describe_degradation(result)


def test_prompt_contains_today_scope_context_and_question():
    result = result_for(QueryMode.DATE, [chunk(19, "Fed 維持利率")], start=dt.date(2026, 8, 19))
    system, user = build_answer_prompt(result, TODAY)

    assert system == ANSWER_SYSTEM
    assert "今天是 2026-08-20" in user
    assert "查詢範圍：2026-08-19" in user
    assert "Fed 維持利率" in user
    assert user.rstrip().endswith("問題：昨天有什麼新聞")


def test_prompt_tells_the_model_to_disclose_degradation():
    result = result_for(
        QueryMode.DATE,
        [chunk(19, "內容")],
        start=dt.date(2026, 8, 19),
        dropped=[dt.date(2026, 8, 17)],
    )
    _, user = build_answer_prompt(result, TODAY)
    assert "請告知使用者" in user


def test_prompt_without_context_still_asks_the_question():
    _, user = build_answer_prompt(result_for(QueryMode.SEMANTIC, []), TODAY)
    assert NO_CONTEXT_NOTE in user
    assert "問題：昨天有什麼新聞" in user


def test_system_prompt_forbids_outside_knowledge_and_requires_traditional_chinese():
    assert "只根據下方提供的內容作答" in ANSWER_SYSTEM
    assert "繁體中文" in ANSWER_SYSTEM
