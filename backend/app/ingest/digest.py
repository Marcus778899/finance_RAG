import datetime as dt
import json
import re
from dataclasses import dataclass, field

from app.providers.base import Generator

MAX_TOPICS = 8
CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)

DIGEST_SYSTEM = (
    "你是金融資訊整理助理。使用者會給你某一天的財經訊息原文，內容中英夾雜。\n"
    "請輸出繁體中文摘要，只根據原文內容，不要補充原文沒有的資訊或數字。\n"
    "只輸出一個 JSON 物件，格式為 "
    '{"summary": "繁體中文摘要", "topics": ["主題1", "主題2"]}。'
)


@dataclass(frozen=True)
class DigestResult:
    summary_zh: str
    key_topics: list[str] = field(default_factory=list)


def build_digest_prompt(day: dt.date, texts: list[str]) -> str:
    body = "\n\n---\n\n".join(text.strip() for text in texts if text.strip())
    return f"日期：{day.isoformat()}\n\n以下是當天的財經訊息原文：\n\n{body}"


def _extract_json_object(raw: str) -> dict | None:
    text = CODE_FENCE.sub("", raw).strip()
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def _clean_topics(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    seen: dict[str, None] = {}
    for item in value:
        if isinstance(item, str) and item.strip():
            seen.setdefault(item.strip(), None)
    return list(seen)[:MAX_TOPICS]


def parse_digest(raw: str) -> DigestResult:
    """模型未照格式輸出時退回把整段當摘要，不要讓 ingest 因為 JSON 壞掉而中斷。"""
    payload = _extract_json_object(raw)
    if payload is None:
        return DigestResult(summary_zh=CODE_FENCE.sub("", raw).strip())

    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return DigestResult(summary_zh=CODE_FENCE.sub("", raw).strip())

    return DigestResult(summary_zh=summary.strip(), key_topics=_clean_topics(payload.get("topics")))


async def generate_digest(generator: Generator, day: dt.date, texts: list[str]) -> DigestResult:
    if not any(text.strip() for text in texts):
        raise ValueError(f"no message text to summarise for {day.isoformat()}")
    raw = await generator.generate(DIGEST_SYSTEM, build_digest_prompt(day, texts))
    return parse_digest(raw)
