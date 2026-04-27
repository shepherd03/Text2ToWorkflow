import os
import sys
import hashlib
import json
import random
import argparse
from collections import Counter
from pathlib import Path

import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.core.schema import DifyNodeType, NodeMappingEvalSample
from src.core.utils import append_to_jsonl, unique_keep_order
from src.dsl_generation.eval_sample_utils import (
    ALLOWED_DIFY_NODE_TYPES,
    build_eval_sample_from_dify_node,
)


DEFAULT_DATASET_PATH = Path("dataset/dataset.jsonl")
DEFAULT_OUTPUT_DIR = Path("generated_data/dsl_generation/node_mapping_eval")
DEFAULT_SAMPLE_MODE = "balanced"

ALLOWED_NODE_TYPES = ALLOWED_DIFY_NODE_TYPES

TARGET_COUNTS = {
    DifyNodeType.llm: 24,
    DifyNodeType.code: 20,
    DifyNodeType.http_request: 18,
    DifyNodeType.template_transform: 18,
    DifyNodeType.tool: 16,
    DifyNodeType.if_else: 14,
    DifyNodeType.iteration: 14,
    DifyNodeType.parameter_extractor: 14,
    DifyNodeType.variable_aggregator: 12,
}


def reset_output_files(output_dir: Path) -> tuple[Path, Path, Path, Path, Path]:
    train_path = output_dir / "train_samples.jsonl"
    valid_path = output_dir / "valid_samples.jsonl"
    test_path = output_dir / "test_samples.jsonl"
    hard_path = output_dir / "hard_samples.jsonl"
    summary_path = output_dir / "split_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in [train_path, valid_path, test_path, hard_path]:
        if path.exists():
            path.unlink()
    return train_path, valid_path, test_path, hard_path, summary_path


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def build_text_signature(action_name: str, description: str) -> str:
    merged = f"{normalize_text(action_name)} || {normalize_text(description)}"
    return hashlib.sha256(merged.encode("utf-8")).hexdigest()


def infer_inputs(node_data: dict) -> list[str]:
    collected: list[str] = []

    for variable in node_data.get("variables", []):
        if isinstance(variable, dict):
            name = variable.get("variable") or variable.get("label")
            if name:
                collected.append(str(name))
        elif isinstance(variable, (list, tuple)) and variable:
            collected.append(str(variable[-1]))

    has_variable_bindings = bool(node_data.get("variables"))

    for selector in [node_data.get("iterator_selector"), node_data.get("output_selector")]:
        if isinstance(selector, list) and selector:
            collected.append(str(selector[-1]))

    for key in ["url", "method", "instruction", "query", "template"]:
        if key in node_data:
            collected.append(key)

    if "conditions" in node_data or node_data.get("cases"):
        collected.append("conditions")

    if "iterator_selector" in node_data:
        collected.append("iterator_selector")

    if has_variable_bindings and node_data.get("type") == "variable-aggregator":
        collected.append("variables")

    if "tool_name" in node_data:
        collected.append("tool_name")
    if "provider_id" in node_data:
        collected.append("provider_id")

    for parameter in node_data.get("parameters", []):
        if isinstance(parameter, dict) and parameter.get("name"):
            collected.append(str(parameter["name"]))

    return unique_keep_order([item for item in collected if item])


def infer_outputs(node_data: dict) -> list[str]:
    collected: list[str] = []
    outputs = node_data.get("outputs")
    if isinstance(outputs, dict):
        collected.extend(str(key) for key in outputs.keys())
    elif isinstance(outputs, list):
        for item in outputs:
            if isinstance(item, dict):
                variable = item.get("variable")
                if variable:
                    collected.append(str(variable))

    if node_data.get("type") == "llm":
        collected.append("text")
    if node_data.get("type") == "tool":
        collected.append("result")
    if node_data.get("type") == "http-request":
        collected.append("response")
    if node_data.get("type") == "parameter-extractor":
        collected.append("parameters")
    if node_data.get("type") == "variable-aggregator":
        collected.append("output")

    return unique_keep_order([item for item in collected if item])


def infer_available_resources(node_data: dict) -> list[str]:
    resources: list[str] = []
    for key in ["provider_name", "provider_id", "tool_name", "tool_label"]:
        value = node_data.get(key)
        if value:
            resources.append(str(value))
    return unique_keep_order(resources)


def infer_description(node_data: dict) -> str:
    parts: list[str] = []
    title = node_data.get("title", "")
    desc = node_data.get("desc", "")
    instruction = node_data.get("instruction", "")
    query = node_data.get("query", "")
    template = node_data.get("template", "")
    code = node_data.get("code", "")

    for text in [desc, instruction, query]:
        if text:
            parts.append(str(text))

    prompt_template = node_data.get("prompt_template", [])
    if isinstance(prompt_template, list):
        for item in prompt_template[:2]:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"])[:300])

    if template:
        parts.append(str(template)[:200])
    if code:
        parts.append(str(code)[:200])
    if title and not parts:
        parts.append(str(title))

    return " ".join(part.strip() for part in parts if str(part).strip())


def infer_parent_block_type(node_data: dict) -> str:
    if node_data.get("isInIteration"):
        return "Loop"
    if node_data.get("type") == "if-else":
        return "Conditional"
    if node_data.get("type") == "iteration":
        return "Loop"
    return "Sequential"


def infer_expected_degraded(expected_type: DifyNodeType, inputs: list[str]) -> bool:
    input_set = set(item.lower() for item in inputs)
    if expected_type == DifyNodeType.http_request:
        return not {"url", "method"}.issubset(input_set)
    if expected_type == DifyNodeType.tool:
        return not {"provider_id", "tool_name"}.issubset(input_set)
    if expected_type == DifyNodeType.if_else:
        return "conditions" not in input_set
    return False


def extract_dataset_samples_balanced(
    dataset_path: Path,
    limit_seed: int = 42,
) -> list[NodeMappingEvalSample]:
    random.seed(limit_seed)
    collected: dict[DifyNodeType, list[NodeMappingEvalSample]] = {
        node_type: [] for node_type in TARGET_COUNTS
    }

    for raw_line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        item = json.loads(raw_line)
        dsl = yaml.safe_load(item["dsl"])
        nodes = dsl.get("workflow", {}).get("graph", {}).get("nodes", [])

        for node in nodes:
            node_id = str(node.get("id", ""))
            node_data = node.get("data", {})
            raw_type = node_data.get("type")
            if raw_type not in ALLOWED_NODE_TYPES:
                continue

            mapped_type = ALLOWED_NODE_TYPES[raw_type]
            if len(collected[mapped_type]) >= TARGET_COUNTS[mapped_type]:
                continue

            sample = build_eval_sample_from_dify_node(
                workflow_id=item["id"],
                source_node_id=node_id,
                node_data=node_data,
                source="dataset",
                split="",
                difficulty="standard",
                tags=[raw_type],
            )
            if sample is None:
                continue
            collected[mapped_type].append(sample)

        if all(len(collected[node_type]) >= target for node_type, target in TARGET_COUNTS.items()):
            break

    samples: list[NodeMappingEvalSample] = []
    for node_type, items in collected.items():
        if len(items) < TARGET_COUNTS[node_type]:
            raise RuntimeError(
                f"Not enough dataset samples for {node_type.value}: {len(items)} < {TARGET_COUNTS[node_type]}"
            )
        samples.extend(items[: TARGET_COUNTS[node_type]])
    return samples


def extract_dataset_samples_full(dataset_path: Path) -> list[NodeMappingEvalSample]:
    samples: list[NodeMappingEvalSample] = []
    seen_sample_ids: set[str] = set()

    for raw_line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        item = json.loads(raw_line)
        dsl = yaml.safe_load(item["dsl"])
        nodes = dsl.get("workflow", {}).get("graph", {}).get("nodes", [])

        for node in nodes:
            node_id = str(node.get("id", ""))
            node_data = node.get("data", {})
            raw_type = node_data.get("type")
            if raw_type not in ALLOWED_NODE_TYPES:
                continue

            sample = build_eval_sample_from_dify_node(
                workflow_id=item["id"],
                source_node_id=node_id,
                node_data=node_data,
                source="dataset",
                split="",
                difficulty="standard",
                tags=[raw_type],
            )
            if sample is None or sample.sample_id in seen_sample_ids:
                continue
            seen_sample_ids.add(sample.sample_id)
            samples.append(sample)

    if not samples:
        raise RuntimeError(f"No supported node mapping samples found in dataset: {dataset_path}")
    return samples


def split_samples(samples: list[NodeMappingEvalSample]) -> tuple[list[NodeMappingEvalSample], list[NodeMappingEvalSample], list[NodeMappingEvalSample]]:
    grouped: dict[DifyNodeType, list[NodeMappingEvalSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.expected_node_type, []).append(sample)

    train: list[NodeMappingEvalSample] = []
    valid: list[NodeMappingEvalSample] = []
    test: list[NodeMappingEvalSample] = []

    for node_type, items in grouped.items():
        items = sorted(items, key=lambda item: item.sample_id)
        total = len(items)
        train_end = max(1, int(total * 0.6))
        valid_end = max(train_end + 1, int(total * 0.8))

        for item in items[:train_end]:
            train.append(item.model_copy(update={"split": "train"}))
        for item in items[train_end:valid_end]:
            valid.append(item.model_copy(update={"split": "valid"}))
        for item in items[valid_end:]:
            test.append(item.model_copy(update={"split": "test"}))

    return train, valid, test


def manual_hard_case(
    sample_id: str,
    expected_node_type: DifyNodeType,
    action_name: str,
    description: str,
    inputs: list[str],
    outputs: list[str],
    available_resources: list[str] | None = None,
    expected_degraded: bool = False,
    tags: list[str] | None = None,
    parent_block_type: str = "Sequential",
) -> NodeMappingEvalSample:
    return NodeMappingEvalSample(
        sample_id=sample_id,
        workflow_id="hard_manual",
        source="hard",
        split="hard",
        expected_node_type=expected_node_type,
        action_name=action_name,
        description=description,
        inputs=inputs,
        outputs=outputs,
        available_resources=available_resources or [],
        expected_degraded=expected_degraded,
        difficulty="hard",
        tags=tags or [],
        parent_block_type=parent_block_type,
        text_signature=build_text_signature(action_name, description),
    )


def build_hard_samples(test_samples: list[NodeMappingEvalSample]) -> list[NodeMappingEvalSample]:
    hard_samples: list[NodeMappingEvalSample] = []

    for sample in test_samples:
        hard_samples.append(
            sample.model_copy(
                update={
                    "sample_id": f"{sample.sample_id}::hard",
                    "source": "hard",
                    "split": "hard",
                    "difficulty": "hard",
                    "tags": sample.tags + ["unseen_rewrite"],
                    "action_name": f"hard::{sample.action_name}",
                    "description": f"Please handle this task variant: {sample.description}"[:600],
                    "text_signature": build_text_signature(
                        f"hard::{sample.action_name}",
                        f"Please handle this task variant: {sample.description}",
                    ),
                }
            )
        )

    extra_cases = [
        manual_hard_case(
            sample_id="hard::tool_missing",
            expected_node_type=DifyNodeType.code,
            action_name="call internal plugin for search",
            description="use external search integration for lookup but only provide query text",
            inputs=["query"],
            outputs=["result"],
            available_resources=["search_service"],
            expected_degraded=True,
            tags=["missing_params", "tool_vs_http"],
        ),
        manual_hard_case(
            sample_id="hard::http_missing",
            expected_node_type=DifyNodeType.code,
            action_name="query partner gateway",
            description="connect to partner api and return the result payload",
            inputs=["payload"],
            outputs=["result"],
            expected_degraded=True,
            tags=["missing_params", "http_vs_tool"],
        ),
        manual_hard_case(
            sample_id="hard::template_llm_conflict",
            expected_node_type=DifyNodeType.template_transform,
            action_name="assemble final notice",
            description="render the final message with a fixed template and placeholders",
            inputs=["template", "title", "body"],
            outputs=["final_notice"],
            tags=["neighbor_conflict", "template_vs_llm"],
        ),
        manual_hard_case(
            sample_id="hard::llm_template_conflict",
            expected_node_type=DifyNodeType.llm,
            action_name="polish user-facing answer",
            description="rewrite, explain, and improve the final response for the user",
            inputs=["draft_text"],
            outputs=["final_answer"],
            tags=["neighbor_conflict", "llm_vs_template"],
        ),
        manual_hard_case(
            sample_id="hard::if_without_conditions",
            expected_node_type=DifyNodeType.code,
            action_name="branch by risk level",
            description="check user risk and route to different branches",
            inputs=["risk_level"],
            outputs=["branch_result"],
            expected_degraded=True,
            tags=["missing_params", "if_else"],
        ),
        manual_hard_case(
            sample_id="hard::zh_if_else_longtail",
            expected_node_type=DifyNodeType.if_else,
            action_name="按风险分数分派处理路径",
            description="先比较风险评分和阈值，再决定是交给人工复核还是直接自动通过",
            inputs=["conditions", "risk_score", "threshold"],
            outputs=["route_result"],
            tags=["zh_long_tail", "if_else", "unseen_rewrite"],
        ),
        manual_hard_case(
            sample_id="hard::zh_iteration_longtail",
            expected_node_type=DifyNodeType.iteration,
            action_name="逐条处理客户留言",
            description="把客户留言列表一条一条拿出来做意图判断和摘要整理",
            inputs=["items", "customer_messages"],
            outputs=["processed_messages"],
            tags=["zh_long_tail", "iteration", "unseen_rewrite"],
        ),
        manual_hard_case(
            sample_id="hard::zh_parameter_extractor_longtail",
            expected_node_type=DifyNodeType.parameter_extractor,
            action_name="抽取投诉里的关键槽位",
            description="从用户投诉文本里抠出城市、日期、产品型号和联系电话",
            inputs=["complaint_text"],
            outputs=["parameters"],
            tags=["zh_long_tail", "parameter_extractor", "entity_slot"],
        ),
        manual_hard_case(
            sample_id="hard::zh_variable_aggregator_longtail",
            expected_node_type=DifyNodeType.variable_aggregator,
            action_name="汇总多路审核结论",
            description="把多个分支返回的审核意见和风险分统一收拢成最终结论",
            inputs=["branch_a_result", "branch_b_result", "branch_c_result"],
            outputs=["final_decision"],
            tags=["zh_long_tail", "variable_aggregator", "multi_branch"],
        ),
        manual_hard_case(
            sample_id="hard::parameter_vs_llm_rewrite",
            expected_node_type=DifyNodeType.parameter_extractor,
            action_name="label the incoming request fields",
            description="identify the city, time window, product name, and intent from the message",
            inputs=["message_text"],
            outputs=["parameters"],
            tags=["neighbor_conflict", "parameter_extractor", "llm_vs_extractor"],
        ),
        manual_hard_case(
            sample_id="hard::aggregator_vs_code_rewrite",
            expected_node_type=DifyNodeType.variable_aggregator,
            action_name="combine branch outputs into one decision packet",
            description="merge the branch level scores, explanations, and recommendations into one payload",
            inputs=["score_a", "score_b", "reason_a", "reason_b"],
            outputs=["decision_packet"],
            tags=["neighbor_conflict", "variable_aggregator", "aggregator_vs_code"],
        ),
        manual_hard_case(
            sample_id="hard::iteration_missing_items",
            expected_node_type=DifyNodeType.code,
            action_name="batch-handle uploaded materials",
            description="walk through the materials and process each one, but the item collection is not given",
            inputs=["upload_summary"],
            outputs=["batch_result"],
            expected_degraded=True,
            tags=["missing_params", "iteration", "zh_long_tail"],
        ),
    ]
    hard_samples.extend(extra_cases)
    return hard_samples


def dump_jsonl(path: Path, samples: list[NodeMappingEvalSample]) -> None:
    for sample in samples:
        append_to_jsonl(str(path), sample)


def build_summary(
    train: list[NodeMappingEvalSample],
    valid: list[NodeMappingEvalSample],
    test: list[NodeMappingEvalSample],
    hard: list[NodeMappingEvalSample],
) -> dict:
    def count_by_label(samples: list[NodeMappingEvalSample]) -> dict[str, int]:
        counter = Counter(sample.expected_node_type.value for sample in samples)
        return dict(sorted(counter.items()))

    return {
        "train_count": len(train),
        "valid_count": len(valid),
        "test_count": len(test),
        "hard_count": len(hard),
        "train_distribution": count_by_label(train),
        "valid_distribution": count_by_label(valid),
        "test_distribution": count_by_label(test),
        "hard_distribution": count_by_label(hard),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare node mapping evaluation samples.")
    parser.add_argument(
        "--dataset-path",
        default=str(DEFAULT_DATASET_PATH),
        help="Input dataset JSONL path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for split JSONL files.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["balanced", "full"],
        default=DEFAULT_SAMPLE_MODE,
        help="balanced keeps the original capped per-label sample construction; full uses all supported nodes.",
    )
    parser.add_argument(
        "--disable-hard",
        action="store_true",
        help="Do not build synthetic hard samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    train_path, valid_path, test_path, hard_path, summary_path = reset_output_files(output_dir)

    if args.sample_mode == "full":
        base_samples = extract_dataset_samples_full(dataset_path)
    else:
        base_samples = extract_dataset_samples_balanced(dataset_path)
    train_samples, valid_samples, test_samples = split_samples(base_samples)
    hard_samples = [] if args.disable_hard else build_hard_samples(test_samples)

    dump_jsonl(train_path, train_samples)
    dump_jsonl(valid_path, valid_samples)
    dump_jsonl(test_path, test_samples)
    dump_jsonl(hard_path, hard_samples)

    summary = build_summary(train_samples, valid_samples, test_samples, hard_samples)
    summary["dataset_path"] = str(dataset_path)
    summary["output_dir"] = str(output_dir)
    summary["sample_mode"] = args.sample_mode
    summary["hard_enabled"] = not args.disable_hard
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
