import math

CJK_RANGES = (
    (0x3000, 0x303F),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0xFF00, 0xFFEF),
)

CHARS_PER_LATIN_TOKEN = 4


def is_cjk(char: str) -> bool:
    code = ord(char)
    return any(start <= code <= end for start, end in CJK_RANGES)


def estimate_tokens(text: str) -> int:
    """粗估 token 數，刻意高估以免低估後被 num_ctx 靜默截斷。"""
    cjk = sum(1 for char in text if is_cjk(char))
    return cjk + math.ceil((len(text) - cjk) / CHARS_PER_LATIN_TOKEN)
