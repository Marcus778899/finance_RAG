import pytest

from app.ingest.chunker import (
    DEFAULT_MAX_TOKENS,
    chunk_text,
    join_sentences,
    split_blocks,
    split_sentences,
)
from app.tokens import estimate_tokens


def test_split_blocks_drops_blank_runs():
    assert split_blocks("a\n\n\n  \n\nb\n") == ["a", "b"]


def test_split_blocks_keeps_single_newlines_together():
    assert split_blocks("- 一\n- 二") == ["- 一\n- 二"]


def test_split_sentences_handles_chinese_punctuation():
    assert split_sentences("美股收紅。台積電上漲！輝達呢？") == [
        "美股收紅。",
        "台積電上漲！",
        "輝達呢？",
    ]


def test_split_sentences_does_not_break_decimals():
    assert split_sentences("Fed held at 4.25-4.50% today.") == ["Fed held at 4.25-4.50% today."]


def test_join_sentences_omits_space_between_chinese():
    assert join_sentences(["美股收紅。", "台積電上漲。"]) == "美股收紅。台積電上漲。"


def test_join_sentences_keeps_space_between_english():
    assert join_sentences(["Fed held.", "Stocks rose."]) == "Fed held. Stocks rose."


def test_join_sentences_on_empty_input():
    assert join_sentences([]) == ""


def test_empty_text_yields_no_chunks():
    assert chunk_text("   \n\n  ") == []


def test_short_text_stays_one_chunk():
    chunks = chunk_text("美股收紅。")
    assert len(chunks) == 1
    assert chunks[0].content == "美股收紅。"
    assert chunks[0].token_count == estimate_tokens("美股收紅。")


def test_blocks_are_packed_up_to_the_limit():
    text = "\n\n".join(["美股收紅。"] * 10)
    chunks = chunk_text(text, max_tokens=20)
    assert len(chunks) > 1
    assert all(chunk.token_count <= 20 for chunk in chunks)


def test_natural_boundaries_are_not_overlapped():
    text = "第一則新聞。\n\n第二則新聞。\n\n第三則新聞。"
    chunks = chunk_text(text, max_tokens=7)
    assert [chunk.content for chunk in chunks] == ["第一則新聞。", "第二則新聞。", "第三則新聞。"]


def test_force_split_block_carries_overlap():
    text = "甲新聞內容。乙新聞內容。丙新聞內容。丁新聞內容。"
    chunks = chunk_text(text, max_tokens=14, overlap_sentences=1)
    assert len(chunks) > 1
    assert chunks[1].content.startswith("乙新聞內容。")


def test_overlap_can_be_disabled():
    text = "甲新聞內容。乙新聞內容。丙新聞內容。丁新聞內容。"
    chunks = chunk_text(text, max_tokens=14, overlap_sentences=0)
    assert chunks[1].content.startswith("丙新聞內容。")


def test_sentence_longer_than_limit_is_hard_split():
    chunks = chunk_text("美" * 100, max_tokens=20)
    assert len(chunks) > 1
    assert "".join(chunk.content for chunk in chunks) == "美" * 100


def test_bullet_list_stays_intact_when_it_fits():
    text = "摘要如下。\n\n- 台積電營收成長\n- 聯發科法說會"
    chunks = chunk_text(text, max_tokens=DEFAULT_MAX_TOKENS)
    assert len(chunks) == 1
    assert "- 台積電營收成長\n- 聯發科法說會" in chunks[0].content


def test_no_content_is_lost_for_realistic_daily_volume():
    text = "\n\n".join(f"第{i}則：聯準會維持利率不變，美股收紅。" * 8 for i in range(20))
    chunks = chunk_text(text)
    assert all(chunk.token_count <= DEFAULT_MAX_TOKENS for chunk in chunks)
    joined = "".join(chunk.content for chunk in chunks)
    for i in range(20):
        assert f"第{i}則" in joined


@pytest.mark.parametrize("max_tokens", [10, 50, 200, 600])
def test_every_chunk_respects_the_limit(max_tokens):
    text = "\n\n".join(["聯準會維持利率不變，美股收紅。" * 5] * 10)
    chunks = chunk_text(text, max_tokens=max_tokens)
    assert chunks
    assert all(chunk.token_count <= max_tokens for chunk in chunks)
