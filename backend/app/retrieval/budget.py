import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol


class Budgetable(Protocol):
    token_count: int
    day: dt.date


@dataclass(frozen=True)
class BudgetOutcome[Item]:
    kept: list[Item]
    dropped_days: list[dt.date] = field(default_factory=list)
    truncated_day: dt.date | None = None

    @property
    def degraded(self) -> bool:
        return bool(self.dropped_days) or self.truncated_day is not None


def group_by_day[T: Budgetable](items: Sequence[T]) -> dict[dt.date, list[T]]:
    grouped: dict[dt.date, list[T]] = {}
    for item in items:
        grouped.setdefault(item.day, []).append(item)
    return grouped


def fit_within_budget[T: Budgetable](items: Sequence[T], budget: int) -> BudgetOutcome[T]:
    """保留最靠近今天的連續區間。最新一天一定留，放不下就在該天內截短。"""
    if budget <= 0:
        return BudgetOutcome(kept=[], dropped_days=sorted({item.day for item in items}))

    grouped = group_by_day(items)
    if not grouped:
        return BudgetOutcome(kept=[])

    days_newest_first = sorted(grouped, reverse=True)
    newest, *older = days_newest_first

    def cost(day: dt.date) -> int:
        return sum(item.token_count for item in grouped[day])

    if cost(newest) > budget:
        partial: list[T] = []
        used = 0
        for item in grouped[newest]:
            if used + item.token_count > budget:
                break
            partial.append(item)
            used += item.token_count
        return BudgetOutcome(kept=partial, dropped_days=sorted(older), truncated_day=newest)

    kept_days = [newest]
    used = cost(newest)
    for day in older:
        if used + cost(day) > budget:
            break
        kept_days.append(day)
        used += cost(day)

    kept = [item for day in sorted(kept_days) for item in grouped[day]]
    dropped = sorted(set(days_newest_first) - set(kept_days))
    return BudgetOutcome(kept=kept, dropped_days=dropped)
