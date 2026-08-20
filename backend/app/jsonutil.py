import json
import re

CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def strip_code_fence(raw: str) -> str:
    return CODE_FENCE.sub("", raw).strip()


def extract_json_object(raw: str) -> dict | None:
    """從可能夾雜前後文的模型輸出中取出第一個完整的 JSON 物件。"""
    text = strip_code_fence(raw)
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
