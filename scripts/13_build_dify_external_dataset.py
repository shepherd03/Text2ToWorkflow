import json
import os
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError

import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.core.config import load_settings
from src.core.llm_client import DeepSeekClient
from src.core.utils import append_to_jsonl
from src.dsl_generation.dify_external_dataset_utils import (
    build_source_metadata,
    build_workflow_overlap_keys,
    collect_repo_raw_file_entries,
    collect_gist_urls,
    collect_issue_ids,
    extract_dsl_blocks_from_issue_html,
    load_dataset_overlap_keys,
    load_gist_page,
    load_issue_page,
    load_raw_gist_file,
    load_raw_repo_file,
    maybe_parse_dify_dsl,
    extract_raw_urls_from_gist_html,
)
from src.dsl_generation.eval_sample_utils import infer_description_from_dify_node


DATASET_PATH = Path("dataset/dataset.jsonl")
OUTPUT_DIR = Path("generated_data/dify_external_dataset")
OUTPUT_DATASET_PATH = OUTPUT_DIR / "dataset.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
ISSUE_CACHE_DIR = OUTPUT_DIR / "cache" / "issues"
GIST_CACHE_DIR = OUTPUT_DIR / "cache" / "gists"
GIST_RAW_CACHE_DIR = OUTPUT_DIR / "cache" / "gist_raw"
REPO_TREE_CACHE_DIR = OUTPUT_DIR / "cache" / "repo_trees"

TARGET_WORKFLOW_COUNT = 120
MAX_ISSUES = 200
MAX_GISTS = 240
MAX_REPO_FILES = 320
MIN_SUPPORTED_NODE_KEYS = 1
GIST_SEARCH_PAGES = [1, 2, 3, 4, 5, 6, 7, 8]
ISSUE_SEARCH_PAGES = [1, 2, 3]
WORKFLOW_NOVEL_RATIO_THRESHOLD = 0.35

INSTRUCTION_SYSTEM_PROMPT = """
你是中文数据集构建助手。

请根据给定的工作流摘要，生成一条“最终用户会怎么描述自己需求”的中文 instruction。

风格要求：
1. 必须是中文。
2. 必须是一句话。
3. 尽量使用“我想……”或“帮我……”开头。
4. 长度控制在 18 到 70 个汉字之间。

内容要求：
1. 只描述任务目标，不描述实现步骤。
2. 不要出现平台、节点、DSL、workflow、JSON、prompt、模型名、工具名。
3. 不要复述输入提示，不要解释你在做什么。
4. 不要输出英文。

输出格式：
只输出 JSON，对象格式固定为 {"instruction": "..."}。

参考风格：
- 我想做一个能根据我的问题，自动从文档里找答案并回复的聊天工具。
- 我想把中文翻译成英文，先翻译一遍，再润色得更自然。
- 我想做个能自动生成完整教程的流程，只要输入主题就行。
""".strip()


def reset_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ISSUE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    GIST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    GIST_RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPO_TREE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for path in [OUTPUT_DATASET_PATH, SUMMARY_PATH, MANIFEST_PATH]:
        if path.exists():
            path.unlink()


def generate_instruction(client: DeepSeekClient, dsl_text: str, source_url: str) -> str:
    dsl = yaml.safe_load(dsl_text)
    graph = dsl.get("workflow", {}).get("graph", {})
    nodes = graph.get("nodes", [])
    node_summaries: list[str] = []
    for node in nodes[:8]:
        node_data = node.get("data", {})
        raw_type = node_data.get("type", "")
        if raw_type not in {
            "llm",
            "tool",
            "http-request",
            "code",
            "template-transform",
            "parameter-extractor",
            "variable-aggregator",
            "if-else",
            "iteration",
        }:
            continue
        summary = infer_description_from_dify_node(node_data, include_title_as_fallback=True)
        node_summaries.append(
            f"- type={raw_type}; title={str(node_data.get('title', ''))[:80]}; summary={summary[:220]}"
        )

    user_prompt = (
        "请根据下面这段 Dify DSL 的结构摘要生成一条中文 instruction。\n"
        f"source_url: {source_url}\n"
        "workflow_name:\n"
        f"{str(dsl.get('app', {}).get('name', ''))[:200]}\n"
        "workflow_description:\n"
        f"{str(dsl.get('app', {}).get('description', ''))[:300]}\n"
        "node_summaries:\n"
        f"{chr(10).join(node_summaries[:8])}\n"
    )
    response = client.chat_json(
        system_prompt=INSTRUCTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.2,
    )
    instruction = str(response.get("instruction", "")).strip()
    if not instruction:
        raise ValueError("LLM 未返回 instruction")
    if not is_valid_instruction(instruction):
        raise ValueError(f"LLM 返回的 instruction 不合规: {instruction[:120]}")
    return instruction


def is_valid_instruction(instruction: str) -> bool:
    text = instruction.strip()
    if len(text) < 8 or len(text) > 120:
        return False
    if re.search(r"[A-Za-z]{6,}", text):
        return False
    invalid_markers = [
        "dify",
        "workflow",
        "llm",
        "tool",
        "json",
        "dsl",
        "节点",
        "工作流",
        "输出",
        "prompt",
    ]
    lowered = text.lower()
    if any(marker in lowered for marker in invalid_markers):
        return False
    return True


def build_dataset_record(
    *,
    record_id: str,
    instruction: str,
    dsl_text: str,
    source_type: str,
    source_url: str,
) -> dict:
    metadata = build_source_metadata(
        source_type=source_type,
        source_url=source_url,
        dsl_text=dsl_text,
    )
    metadata["id"] = record_id
    return {
        "id": record_id,
        "instruction": instruction,
        "dsl": dsl_text,
        "metadata": metadata,
    }


def collect_issue_workflows(
    *,
    dataset_overlap_keys: set[str],
    max_issues: int = MAX_ISSUES,
) -> list[dict]:
    collected: list[dict] = []
    seen_overlap_sets: set[tuple[str, ...]] = set()
    issue_ids = collect_issue_ids(search_pages=ISSUE_SEARCH_PAGES)

    for issue_id in issue_ids[:max_issues]:
        try:
            issue_html = load_issue_page(issue_id, ISSUE_CACHE_DIR)
        except (HTTPError, URLError):
            continue

        for block_index, dsl_text in enumerate(extract_dsl_blocks_from_issue_html(issue_html)):
            overlap_keys = build_workflow_overlap_keys(dsl_text)
            if not overlap_keys:
                continue
            if len(overlap_keys) < MIN_SUPPORTED_NODE_KEYS:
                continue
            novel_ratio = 1.0
            if overlap_keys:
                novel_ratio = (len(overlap_keys - dataset_overlap_keys) / len(overlap_keys))
            if novel_ratio < WORKFLOW_NOVEL_RATIO_THRESHOLD:
                continue

            overlap_signature = tuple(sorted(overlap_keys))
            if overlap_signature in seen_overlap_sets:
                continue
            seen_overlap_sets.add(overlap_signature)

            collected.append(
                {
                    "source_type": "github_issue",
                    "source_url": f"https://github.com/langgenius/dify/issues/{issue_id}#dsl-{block_index}",
                    "dsl": dsl_text,
                    "overlap_keys": overlap_keys,
                    "novel_ratio": novel_ratio,
                }
            )
    return collected


def collect_gist_workflows(
    *,
    dataset_overlap_keys: set[str],
    max_gists: int = MAX_GISTS,
) -> list[dict]:
    collected: list[dict] = []
    seen_overlap_sets: set[tuple[str, ...]] = set()
    gist_urls = collect_gist_urls(
        search_pages=GIST_SEARCH_PAGES,
        max_results=max_gists,
    )

    for gist_url in gist_urls:
        try:
            gist_html = load_gist_page(gist_url, GIST_CACHE_DIR)
        except (HTTPError, URLError):
            continue

        raw_urls = extract_raw_urls_from_gist_html(gist_html)
        for raw_url in raw_urls:
            try:
                raw_text = load_raw_gist_file(raw_url, GIST_RAW_CACHE_DIR)
            except (HTTPError, URLError):
                continue

            dsl_text = maybe_parse_dify_dsl(raw_text)
            if dsl_text is None:
                continue

            overlap_keys = build_workflow_overlap_keys(dsl_text)
            if not overlap_keys:
                continue
            if len(overlap_keys) < MIN_SUPPORTED_NODE_KEYS:
                continue
            novel_ratio = 1.0
            if overlap_keys:
                novel_ratio = (len(overlap_keys - dataset_overlap_keys) / len(overlap_keys))
            if novel_ratio < WORKFLOW_NOVEL_RATIO_THRESHOLD:
                continue

            overlap_signature = tuple(sorted(overlap_keys))
            if overlap_signature in seen_overlap_sets:
                continue
            seen_overlap_sets.add(overlap_signature)

            collected.append(
                {
                    "source_type": "gist_raw",
                    "source_url": raw_url,
                    "dsl": dsl_text,
                    "overlap_keys": overlap_keys,
                    "novel_ratio": novel_ratio,
                }
            )
    return collected


def collect_repo_workflows(
    *,
    dataset_overlap_keys: set[str],
    max_repo_files: int = MAX_REPO_FILES,
) -> list[dict]:
    collected: list[dict] = []
    seen_overlap_sets: set[tuple[str, ...]] = set()
    raw_entries = collect_repo_raw_file_entries(
        cache_dir=REPO_TREE_CACHE_DIR,
        max_results=max_repo_files,
    )

    for entry in raw_entries:
        raw_url = entry["source_url"]
        try:
            raw_text = load_raw_repo_file(raw_url, GIST_RAW_CACHE_DIR)
        except (HTTPError, URLError):
            continue

        dsl_text = maybe_parse_dify_dsl(raw_text)
        if dsl_text is None:
            continue

        overlap_keys = build_workflow_overlap_keys(dsl_text)
        if not overlap_keys:
            continue
        if len(overlap_keys) < MIN_SUPPORTED_NODE_KEYS:
            continue
        novel_ratio = len(overlap_keys - dataset_overlap_keys) / len(overlap_keys)
        if novel_ratio < WORKFLOW_NOVEL_RATIO_THRESHOLD:
            continue

        overlap_signature = tuple(sorted(overlap_keys))
        if overlap_signature in seen_overlap_sets:
            continue
        seen_overlap_sets.add(overlap_signature)

        collected.append(
            {
                "source_type": "github_repo_raw",
                "source_url": raw_url,
                "dsl": dsl_text,
                "overlap_keys": overlap_keys,
                "novel_ratio": novel_ratio,
                "repo_id": entry["repo_id"],
                "repo_path": entry["path"],
            }
        )
    return collected


def main() -> None:
    reset_outputs()

    settings = load_settings()
    if not settings.llm_enabled:
        raise RuntimeError("当前未配置 DeepSeek API，无法执行 DSL -> instruction 数据集构建。")

    client = DeepSeekClient(settings)
    dataset_overlap_keys = load_dataset_overlap_keys(DATASET_PATH)

    gist_workflows = collect_gist_workflows(
        dataset_overlap_keys=dataset_overlap_keys,
        max_gists=MAX_GISTS,
    )
    repo_workflows = collect_repo_workflows(
        dataset_overlap_keys=dataset_overlap_keys,
        max_repo_files=MAX_REPO_FILES,
    )
    issue_workflows = collect_issue_workflows(
        dataset_overlap_keys=dataset_overlap_keys,
        max_issues=MAX_ISSUES,
    )
    candidates = sorted(
        repo_workflows + gist_workflows + issue_workflows,
        key=lambda item: (
            -len(item.get("overlap_keys", [])),
            -float(item.get("novel_ratio", 0.0)),
            item.get("source_type", ""),
            item.get("source_url", ""),
        ),
    )

    records: list[dict] = []
    manifest = {
        "issue_workflow_candidates": len(issue_workflows),
        "gist_workflow_candidates": len(gist_workflows),
        "repo_workflow_candidates": len(repo_workflows),
        "selected_sources": [],
    }

    for index, item in enumerate(candidates, start=1):
        if len(records) >= TARGET_WORKFLOW_COUNT:
            break

        record_id = f"dify_ext_{index:04d}"
        try:
            instruction = generate_instruction(client, item["dsl"], item["source_url"])
        except Exception as exc:
            manifest["selected_sources"].append(
                {
                    "id": record_id,
                    "source_url": item["source_url"],
                    "source_type": item["source_type"],
                    "status": "instruction_failed",
                    "error": str(exc),
                }
            )
            continue

        record = build_dataset_record(
            record_id=record_id,
            instruction=instruction,
            dsl_text=item["dsl"],
            source_type=item["source_type"],
            source_url=item["source_url"],
        )
        append_to_jsonl(str(OUTPUT_DATASET_PATH), record)
        records.append(record)
        manifest["selected_sources"].append(
                {
                    "id": record_id,
                    "source_url": item["source_url"],
                    "source_type": item["source_type"],
                    "status": "accepted",
                    "novel_ratio": item.get("novel_ratio"),
                    "node_count": record["metadata"]["node_count"],
                    "edge_count": record["metadata"]["edge_count"],
                    "supported_node_type_counts": record["metadata"]["supported_node_type_counts"],
                    "repo_id": item.get("repo_id"),
                    "repo_path": item.get("repo_path"),
            }
        )

    summary = {
        "output_dataset_path": str(OUTPUT_DATASET_PATH),
        "workflow_count": len(records),
        "target_workflow_count": TARGET_WORKFLOW_COUNT,
        "sources": {
            "github_issue": sum(1 for record in records if record["metadata"]["source"] == "github_issue"),
            "gist_raw": sum(1 for record in records if record["metadata"]["source"] == "gist_raw"),
            "github_repo_raw": sum(
                1 for record in records if record["metadata"]["source"] == "github_repo_raw"
            ),
        },
        "total_supported_nodes": sum(
            sum(record["metadata"]["supported_node_type_counts"].values())
            for record in records
        ),
    }

    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
