import json
import re
from typing import Any


def safe_json_extract(raw: str) -> Any:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", text)
        if not match:
            raise
        return json.loads(match.group(0))


def unique_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def format_json_for_readability(data: Any) -> str:
    """
    Format JSON data into a highly readable string format with 2-space indentation.
    Ensures ASCII characters are not escaped for better readability of non-English text.
    """
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    return json.dumps(data, ensure_ascii=False, indent=2)

def append_to_jsonl(file_path: str, data: Any) -> None:
    import os
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

