import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_RULEBOOK_PATH = Path(__file__).resolve().parent.parent / "configs" / "rules"
REQUIRED_SECTIONS = {
    "preprocessor",
    "action_extractor",
    "resource_extractor",
    "control_intent_extractor",
    "variable_extractor",
}


def resolve_rulebook_path(rule_path: str | Path | None = None) -> Path:
    if rule_path:
        return Path(rule_path).resolve()
    env_path = os.getenv("UTR_RULEBOOK_PATH", "").strip()
    if env_path:
        return Path(env_path).resolve()
    return DEFAULT_RULEBOOK_PATH


@lru_cache(maxsize=8)
def _load_rulebook_by_path(path_str: str) -> dict[str, Any]:
    rule_path = Path(path_str)
    if rule_path.is_dir():
        rules = _load_rulebook_from_dir(rule_path)
    else:
        with rule_path.open("r", encoding="utf-8") as f:
            rules = json.load(f)
    missing_sections = [key for key in REQUIRED_SECTIONS if key not in rules]
    if missing_sections:
        raise ValueError(f"规则配置缺少必要模块: {missing_sections}")
    return rules


def _load_rulebook_from_dir(rule_dir: Path) -> dict[str, Any]:
    rules: dict[str, Any] = {}
    for json_file in sorted(rule_dir.glob("*.json")):
        section_name = json_file.stem
        with json_file.open("r", encoding="utf-8") as f:
            rules[section_name] = json.load(f)
    return rules


def load_rulebook(rule_path: str | Path | None = None) -> dict[str, Any]:
    path = resolve_rulebook_path(rule_path)
    return _load_rulebook_by_path(str(path))
