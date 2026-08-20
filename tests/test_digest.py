import datetime as dt
import json

import pytest

from app.ingest.digest import (
    DIGEST_SYSTEM,
    MAX_TOPICS,
    DigestResult,
    build_digest_prompt,
    generate_digest,
    parse_digest,
)
from app.providers import FakeGenerator

DAY = dt.date(2026, 8, 19)


def test_prompt_includes_date_and_all_texts():
    prompt = build_digest_prompt(DAY, ["美股收紅", "Fed held rates"])
    assert "2026-08-19" in prompt
    assert "美股收紅" in prompt
    assert "Fed held rates" in prompt


def test_prompt_skips_blank_texts():
    assert "---" not in build_digest_prompt(DAY, ["美股收紅", "   "])


def test_parse_plain_json():
    result = parse_digest('{"summary": "美股收紅。", "topics": ["美股", "Fed"]}')
    assert result == DigestResult(summary_zh="美股收紅。", key_topics=["美股", "Fed"])


def test_parse_strips_code_fence():
    raw = '```json\n{"summary": "美股收紅。", "topics": ["美股"]}\n```'
    assert parse_digest(raw).summary_zh == "美股收紅。"


def test_parse_ignores_text_around_the_object():
    raw = '好的，以下是摘要：\n{"summary": "美股收紅。", "topics": []}\n希望有幫助。'
    assert parse_digest(raw).summary_zh == "美股收紅。"


def test_parse_handles_nested_objects():
    raw = '{"summary": "美股收紅。", "topics": ["美股"], "meta": {"n": 1}}'
    assert parse_digest(raw).key_topics == ["美股"]


def test_parse_falls_back_to_whole_text_when_not_json():
    result = parse_digest("美股收紅，台積電上漲。")
    assert result.summary_zh == "美股收紅，台積電上漲。"
    assert result.key_topics == []


def test_parse_falls_back_on_malformed_json():
    assert parse_digest('{"summary": "美股收紅。",}').summary_zh == '{"summary": "美股收紅。",}'


def test_parse_falls_back_when_summary_is_missing():
    raw = '{"topics": ["美股"]}'
    assert parse_digest(raw).summary_zh == raw


def test_parse_falls_back_when_summary_is_not_a_string():
    raw = '{"summary": 42}'
    assert parse_digest(raw).summary_zh == raw


def test_parse_falls_back_on_unterminated_object():
    raw = '{"summary": "美股收紅。", "topics": ["美股"]'
    assert parse_digest(raw).summary_zh == raw


def test_parse_rejects_non_object_json():
    assert parse_digest("[1, 2, 3]").summary_zh == "[1, 2, 3]"


def test_topics_are_deduplicated_and_capped():
    topics = [f"主題{i}" for i in range(20)] + ["主題0"]
    result = parse_digest(json.dumps({"summary": "x", "topics": topics}))
    assert len(result.key_topics) == MAX_TOPICS
    assert len(set(result.key_topics)) == MAX_TOPICS


def test_topics_drop_non_strings_and_blanks():
    result = parse_digest('{"summary": "x", "topics": ["美股", 1, "  ", null, "Fed"]}')
    assert result.key_topics == ["美股", "Fed"]


def test_topics_default_to_empty_when_not_a_list():
    assert parse_digest('{"summary": "x", "topics": "美股"}').key_topics == []


async def test_generate_digest_uses_the_digest_system_prompt():
    generator = FakeGenerator(['{"summary": "美股收紅。", "topics": ["美股"]}'])
    result = await generate_digest(generator, DAY, ["原文"])

    assert result.summary_zh == "美股收紅。"
    assert generator.prompts[0][0] == DIGEST_SYSTEM


async def test_generate_digest_rejects_a_day_with_no_text():
    with pytest.raises(ValueError, match="no message text"):
        await generate_digest(FakeGenerator(), DAY, ["  ", ""])
