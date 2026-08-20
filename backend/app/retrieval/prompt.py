import datetime as dt
from collections.abc import Sequence

from app.models import ChunkSource
from app.retrieval.budget import group_by_day
from app.retrieval.query_parser import QueryMode
from app.retrieval.retriever import RetrievalResult, RetrievedChunk

ANSWER_SYSTEM = (
    "你是金融資訊助理，回答關於某個 Slack 頻道每日財經摘要的問題。\n"
    "規則：\n"
    "1. 只根據下方提供的內容作答，不要引用其他知識，不要推測。\n"
    "2. 提供的內容中英夾雜，一律用繁體中文回答。\n"
    "3. 數字、公司名、日期必須與原文一致，不確定就不要寫。\n"
    "4. 提到某則消息時標註其日期，例如（08-19）。\n"
    "5. 若提供的內容不足以回答，直接說明沒有相關資料，不要編造。"
)

NO_CONTEXT_NOTE = "（沒有任何符合條件的內容）"


def _format_chunk(chunk: RetrievedChunk) -> str:
    label = "當日摘要" if chunk.source is ChunkSource.DIGEST else "原文"
    author = f" @{chunk.author}" if chunk.author else ""
    return f"[{label}{author}]\n{chunk.content}"


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    if not chunks:
        return NO_CONTEXT_NOTE
    grouped = group_by_day(chunks)
    sections = [
        f"### {day.isoformat()}\n\n" + "\n\n".join(_format_chunk(chunk) for chunk in grouped[day])
        for day in sorted(grouped)
    ]
    return "\n\n".join(sections)


def describe_scope(result: RetrievalResult) -> str:
    plan = result.plan
    if plan.date_range is None:
        return "查詢範圍：全部歷史（依語意相似度挑選）"

    span = f"{plan.date_range.start.isoformat()} 至 {plan.date_range.end.isoformat()}"
    if plan.mode is QueryMode.HYBRID:
        return f"查詢範圍：{span}（區間較長，提供當日摘要與相關片段）"
    return f"查詢範圍：{span}"


def describe_degradation(result: RetrievalResult) -> str | None:
    if not result.degraded:
        return None
    notes = []
    if result.dropped_days:
        days = "、".join(day.isoformat() for day in result.dropped_days)
        notes.append(f"因長度限制未納入這些日期：{days}")
    if result.truncated_day:
        notes.append(f"{result.truncated_day.isoformat()} 的內容過長，只納入前段")
    return "；".join(notes)


def build_answer_prompt(result: RetrievalResult, today: dt.date) -> tuple[str, str]:
    parts = [f"今天是 {today.isoformat()}。", describe_scope(result)]
    if note := describe_degradation(result):
        parts.append(f"注意：{note}。回答時請告知使用者這一點。")
    parts.append(f"---\n\n{format_context(result.chunks)}\n\n---")
    parts.append(f"問題：{result.plan.question}")
    return ANSWER_SYSTEM, "\n\n".join(parts)
