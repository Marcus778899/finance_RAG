import pytest

from app.tokens import estimate_tokens, is_cjk


@pytest.mark.parametrize("char, expected", [("中", True), ("，", True), ("a", False), ("1", False)])
def test_is_cjk(char, expected):
    assert is_cjk(char) is expected


def test_empty_text_is_zero():
    assert estimate_tokens("") == 0


def test_cjk_counts_one_token_per_char():
    assert estimate_tokens("聯準會維持利率") == 7


def test_latin_counts_four_chars_per_token():
    assert estimate_tokens("abcdefgh") == 2


def test_mixed_text_sums_both_parts():
    assert estimate_tokens("美股 SP500 上漲") == estimate_tokens("美股上漲") + estimate_tokens(
        " SP500 "
    )


def test_estimate_does_not_underestimate_typical_daily_volume():
    daily = "聯準會維持利率不變，美股收紅。" * 300
    assert estimate_tokens(daily) >= len(daily) * 0.9
