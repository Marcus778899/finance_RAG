import datetime as dt
from dataclasses import dataclass

from app.retrieval.budget import fit_within_budget, group_by_day


@dataclass
class Item:
    token_count: int
    day: dt.date
    tag: str = ""


def items(*specs: tuple[int, int]) -> list[Item]:
    return [Item(tokens, dt.date(2026, 8, day), f"d{day}") for tokens, day in specs]


def tags(outcome) -> list[str]:
    return [item.tag for item in outcome.kept]


def test_group_by_day_preserves_order_within_a_day():
    grouped = group_by_day([Item(1, dt.date(2026, 8, 19), "a"), Item(1, dt.date(2026, 8, 19), "b")])
    assert [item.tag for item in grouped[dt.date(2026, 8, 19)]] == ["a", "b"]


def test_everything_fits_when_the_budget_is_generous():
    outcome = fit_within_budget(items((100, 17), (100, 18), (100, 19)), 1000)
    assert tags(outcome) == ["d17", "d18", "d19"]
    assert not outcome.degraded


def test_oldest_days_are_dropped_first():
    outcome = fit_within_budget(items((100, 17), (100, 18), (100, 19), (100, 20)), 250)
    assert tags(outcome) == ["d19", "d20"]
    assert outcome.dropped_days == [dt.date(2026, 8, 17), dt.date(2026, 8, 18)]
    assert outcome.degraded


def test_kept_chunks_come_back_in_chronological_order():
    outcome = fit_within_budget(items((100, 20), (100, 19), (100, 18)), 250)
    assert tags(outcome) == ["d19", "d20"]


def test_newest_day_is_truncated_when_it_alone_exceeds_the_budget():
    chunks = [Item(60, dt.date(2026, 8, 20), "a"), Item(60, dt.date(2026, 8, 20), "b")]
    outcome = fit_within_budget([*items((100, 19)), *chunks], 100)

    assert tags(outcome) == ["a"]
    assert outcome.truncated_day == dt.date(2026, 8, 20)
    assert outcome.dropped_days == [dt.date(2026, 8, 19)]
    assert outcome.degraded


def test_truncation_can_keep_nothing_when_the_first_chunk_is_too_big():
    outcome = fit_within_budget(items((500, 20)), 100)
    assert outcome.kept == []
    assert outcome.truncated_day == dt.date(2026, 8, 20)


def test_zero_budget_drops_everything():
    outcome = fit_within_budget(items((10, 19), (10, 20)), 0)
    assert outcome.kept == []
    assert outcome.dropped_days == [dt.date(2026, 8, 19), dt.date(2026, 8, 20)]
    assert outcome.degraded


def test_empty_input_is_not_degraded():
    outcome = fit_within_budget([], 100)
    assert outcome.kept == []
    assert not outcome.degraded


def test_a_day_is_kept_or_dropped_whole():
    day_chunks = [Item(80, dt.date(2026, 8, 19), "a"), Item(80, dt.date(2026, 8, 19), "b")]
    outcome = fit_within_budget([*day_chunks, *items((100, 20))], 200)
    assert tags(outcome) == ["d20"]
    assert outcome.dropped_days == [dt.date(2026, 8, 19)]


def test_newest_day_is_never_skipped_in_favour_of_a_smaller_older_day():
    chunks = [Item(120, dt.date(2026, 8, 20), "new")]
    outcome = fit_within_budget([*items((100, 19)), *chunks], 100)

    assert outcome.truncated_day == dt.date(2026, 8, 20)
    assert outcome.dropped_days == [dt.date(2026, 8, 19)]


def test_kept_window_is_contiguous_from_the_newest_day():
    spread = [*items((90, 15)), *items((200, 16)), *items((90, 17))]
    outcome = fit_within_budget(spread, 200)
    assert tags(outcome) == ["d17"]
    assert outcome.dropped_days == [dt.date(2026, 8, 15), dt.date(2026, 8, 16)]
