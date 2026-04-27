import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError

import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.core.schema import DifyNodeType, NodeMappingEvalSample
from src.core.utils import append_to_jsonl
from src.dsl_generation.dify_external_dataset_utils import (
    collect_issue_ids,
    extract_dsl_blocks_from_issue_html,
    load_dataset_overlap_keys,
    load_issue_page,
)
from src.dsl_generation.eval_sample_utils import (
    ALLOWED_DIFY_NODE_TYPES,
    build_eval_sample_from_dify_node,
    build_node_overlap_key,
)


DATASET_PATH = Path("dataset/dataset.jsonl")
OUTPUT_DIR = Path("generated_data/dsl_generation/dify_external_node_mapping_eval")
SAMPLES_PATH = OUTPUT_DIR / "external_samples.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "external_summary.json"
MANIFEST_PATH = OUTPUT_DIR / "external_manifest.json"
CACHE_DIR = OUTPUT_DIR / "cache"
ISSUE_IDS_PATH = OUTPUT_DIR / "issue_ids.json"

ISSUE_SEARCH_QUERIES = [
    'repo:langgenius/dify "kind: app" "workflow:"',
    'repo:langgenius/dify "DSL content"',
    'repo:langgenius/dify "```yaml" "kind: app"',
    'repo:langgenius/dify "type: if-else" "kind: app"',
    'repo:langgenius/dify "type: code" "kind: app"',
    'repo:langgenius/dify "type: tool" "provider_id:"',
    'repo:langgenius/dify "type: parameter-extractor"',
    'repo:langgenius/dify "type: variable-aggregator"',
]
ISSUE_SEARCH_PAGES = [1, 2, 3]
TARGET_TOTAL = 220
PER_LABEL_CAP = 40
MAX_FETCH_ISSUES = 120


def reset_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for path in [SAMPLES_PATH, SUMMARY_PATH, MANIFEST_PATH, ISSUE_IDS_PATH]:
        if path.exists():
            path.unlink()


def sanitize_external_sample(sample: NodeMappingEvalSample) -> NodeMappingEvalSample:
    lowered = sample.action_name.strip().lower()
    output_names = [item.lower() for item in sample.outputs]
    input_names = [item.lower() for item in sample.inputs]

    if sample.expected_node_type == DifyNodeType.llm and lowered in {
        "llm",
        "chat model",
        "reasoning model",
        "openai chat model",
        "model",
    }:
        sample = sample.model_copy(update={"action_name": ""})
    elif sample.expected_node_type == DifyNodeType.code and lowered in {"code", "script", "代码", "代码执行"}:
        sample = sample.model_copy(update={"action_name": ""})
    elif sample.expected_node_type == DifyNodeType.if_else and lowered in {"if/else", "if", "条件分支"}:
        sample = sample.model_copy(update={"action_name": ""})
    elif sample.expected_node_type == DifyNodeType.iteration and lowered in {"iteration", "loop", "循环"}:
        sample = sample.model_copy(update={"action_name": ""})
    elif sample.expected_node_type == DifyNodeType.parameter_extractor and lowered in {
        "parameter extractor",
        "information extractor",
        "参数提取器",
    }:
        sample = sample.model_copy(update={"action_name": ""})
    elif sample.expected_node_type == DifyNodeType.variable_aggregator and lowered in {
        "variable aggregator",
        "aggregator",
        "变量聚合器",
    }:
        sample = sample.model_copy(update={"action_name": ""})
    elif sample.expected_node_type == DifyNodeType.http_request and lowered in {
        "http request",
        "request",
        "api request",
        "接口请求",
    }:
        sample = sample.model_copy(update={"action_name": ""})
    elif sample.expected_node_type == DifyNodeType.tool and lowered in {"tool", "工具"}:
        sample = sample.model_copy(update={"action_name": ""})
    elif sample.expected_node_type == DifyNodeType.template_transform and lowered in {
        "template transform",
        "template",
        "模板转换",
    }:
        sample = sample.model_copy(update={"action_name": ""})

    # Remove default output label leakage where it would make the sample trivial.
    if sample.expected_node_type == DifyNodeType.llm and output_names == ["text"]:
        sample = sample.model_copy(update={"outputs": []})
    elif sample.expected_node_type == DifyNodeType.code and output_names == ["result"]:
        sample = sample.model_copy(update={"outputs": []})
    elif sample.expected_node_type == DifyNodeType.variable_aggregator and output_names == ["output"]:
        sample = sample.model_copy(update={"outputs": []})
    elif sample.expected_node_type == DifyNodeType.tool and output_names == ["result"]:
        sample = sample.model_copy(update={"outputs": []})
    elif sample.expected_node_type == DifyNodeType.http_request and output_names == ["response"]:
        sample = sample.model_copy(update={"outputs": []})
    elif sample.expected_node_type == DifyNodeType.parameter_extractor and output_names == ["parameters"]:
        sample = sample.model_copy(update={"outputs": []})

    if sample.expected_node_type == DifyNodeType.tool:
        sample = sample.model_copy(
            update={
                "available_resources": [
                    item
                    for item in sample.available_resources
                    if item.lower() not in {"tool_name", "provider_id", "tool_label", "provider_name"}
                ]
            }
        )
    if sample.expected_node_type == DifyNodeType.http_request:
        sample = sample.model_copy(
            update={
                "inputs": [item for item in sample.inputs if item.lower() not in {"url", "method"}]
            }
        )
    if sample.expected_node_type == DifyNodeType.parameter_extractor and "instruction" in input_names:
        sample = sample.model_copy(
            update={"inputs": [item for item in sample.inputs if item.lower() != "instruction"]}
        )

    return sample.model_copy(
        update={
            "text_signature": sample.text_signature
            if sample.action_name or sample.description
            else sample.text_signature,
        }
    )


def collect_external_samples() -> tuple[list[NodeMappingEvalSample], dict[str, list[str]], dict]:
    dataset_overlap_keys = load_dataset_overlap_keys(DATASET_PATH)
    issue_ids = collect_issue_ids(
        search_queries=ISSUE_SEARCH_QUERIES,
        search_pages=ISSUE_SEARCH_PAGES,
    )
    ISSUE_IDS_PATH.write_text(json.dumps(issue_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    per_label: dict[DifyNodeType, list[NodeMappingEvalSample]] = defaultdict(list)
    manifest: dict[str, list[str]] = {}
    seen_signatures: set[str] = set()
    seen_overlap_keys: set[str] = set()

    fetched_issues = 0
    for issue_id in issue_ids:
        if fetched_issues >= MAX_FETCH_ISSUES:
            break

        try:
            issue_html = load_issue_page(issue_id, CACHE_DIR)
        except (HTTPError, URLError):
            continue

        fetched_issues += 1
        blocks = extract_dsl_blocks_from_issue_html(issue_html)
        if not blocks:
            continue

        issue_samples: list[str] = []
        for block_index, dsl_text in enumerate(blocks):
            try:
                dsl = yaml.safe_load(dsl_text)
            except yaml.YAMLError:
                continue

            nodes = dsl.get("workflow", {}).get("graph", {}).get("nodes", [])
            for node in nodes:
                node_id = str(node.get("id", ""))
                node_data = node.get("data", {})
                if node_data.get("type") not in ALLOWED_DIFY_NODE_TYPES:
                    continue

                overlap_key = build_node_overlap_key(node_data)
                if overlap_key in dataset_overlap_keys or overlap_key in seen_overlap_keys:
                    continue

                sample = build_eval_sample_from_dify_node(
                    workflow_id=f"dify_issue::{issue_id}::{block_index}",
                    source_node_id=node_id,
                    node_data=node_data,
                    source="external_dify_issue",
                    split="external",
                    difficulty="external",
                    tags=["dify", "github_issue", f"issue:{issue_id}", str(node_data.get("type", ""))],
                    title_mode="sanitized",
                )
                if sample is None:
                    continue

                sample = sanitize_external_sample(sample)
                label = sample.expected_node_type

                if len(per_label[label]) >= PER_LABEL_CAP:
                    continue
                if sample.text_signature in seen_signatures:
                    continue
                if not sample.action_name and not sample.description.strip():
                    continue

                per_label[label].append(sample)
                seen_signatures.add(sample.text_signature)
                seen_overlap_keys.add(overlap_key)
                issue_samples.append(sample.sample_id)

        if issue_samples:
            manifest[str(issue_id)] = issue_samples

        current_total = sum(len(items) for items in per_label.values())
        if current_total >= TARGET_TOTAL and all(len(items) >= 12 for items in per_label.values()):
            break

    samples: list[NodeMappingEvalSample] = []
    for node_type in sorted(per_label.keys(), key=lambda item: item.value):
        samples.extend(per_label[node_type])

    summary = {
        "issue_count_collected": len(manifest),
        "issue_count_scanned": fetched_issues,
        "sample_count": len(samples),
        "distribution": {
            node_type.value: len(items)
            for node_type, items in sorted(per_label.items(), key=lambda item: item[0].value)
        },
        "target_total": TARGET_TOTAL,
        "per_label_cap": PER_LABEL_CAP,
        "dedup_against_dataset": True,
        "title_sanitized": True,
        "sources": ["langgenius/dify GitHub issues"],
    }
    return samples, manifest, summary


def main() -> None:
    reset_outputs()
    samples, manifest, summary = collect_external_samples()
    for sample in samples:
        append_to_jsonl(str(SAMPLES_PATH), sample)

    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
