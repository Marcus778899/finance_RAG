import re
from dataclasses import dataclass

from app.tokens import estimate_tokens

DEFAULT_MAX_TOKENS = 600
DEFAULT_OVERLAP_SENTENCES = 1

BLANK_LINE = re.compile(r"\n\s*\n+")
SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？；])|(?<=[.!?;])(?=\s)")


@dataclass(frozen=True)
class TextChunk:
    content: str
    token_count: int


def split_blocks(text: str) -> list[str]:
    return [block.strip() for block in BLANK_LINE.split(text.strip()) if block.strip()]


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_BOUNDARY.split(text) if part.strip()]


def join_sentences(parts: list[str]) -> str:
    """中文句子之間不補空格，英文之間補。"""
    if not parts:
        return ""
    out = parts[0]
    for part in parts[1:]:
        separator = " " if (out[-1].isascii() and part[0].isascii()) else ""
        out += separator + part
    return out


def _hard_split(text: str, max_tokens: int) -> list[str]:
    step = max(1, max_tokens // 2)
    return [text[index : index + step] for index in range(0, len(text), step)]


def _split_oversized_block(block: str, max_tokens: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            pieces.append(join_sentences(current))

    for sentence in split_sentences(block):
        if estimate_tokens(sentence) > max_tokens:
            flush()
            current.clear()
            pieces.extend(_hard_split(sentence, max_tokens))
            continue

        candidate = [*current, sentence]
        if estimate_tokens(join_sentences(candidate)) > max_tokens and current:
            flush()
            current = [*current[len(current) - overlap :], sentence] if overlap else [sentence]
        else:
            current = candidate

    flush()
    return pieces


def chunk_text(
    text: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
) -> list[TextChunk]:
    """切成語意完整的片段。只有在硬切開一個語意單位時才加重疊。"""
    contents: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            contents.append("\n\n".join(current))
            current.clear()

    for block in split_blocks(text):
        if estimate_tokens(block) > max_tokens:
            flush()
            contents.extend(_split_oversized_block(block, max_tokens, overlap_sentences))
            continue

        if current and estimate_tokens("\n\n".join([*current, block])) > max_tokens:
            flush()
        current.append(block)

    flush()
    return [
        TextChunk(content=content, token_count=estimate_tokens(content)) for content in contents
    ]
