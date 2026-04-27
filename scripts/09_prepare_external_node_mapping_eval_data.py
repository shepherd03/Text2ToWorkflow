import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.core.schema import DifyNodeType, NodeMappingEvalSample
from src.core.utils import append_to_jsonl, unique_keep_order


OUTPUT_DIR = Path("generated_data/dsl_generation/external_node_mapping_eval")
SAMPLES_PATH = OUTPUT_DIR / "external_samples.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "external_summary.json"
MANIFEST_PATH = OUTPUT_DIR / "external_manifest.json"

SEARCH_BASE_URL = "https://api.n8n.io/templates/search"
WORKFLOW_BASE_URL = "https://api.n8n.io/templates/workflows"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}


TARGET_COUNTS = {
    DifyNodeType.llm: 5,
    DifyNodeType.code: 5,
    DifyNodeType.http_request: 5,
    DifyNodeType.template_transform: 5,
    DifyNodeType.tool: 5,
    DifyNodeType.if_else: 5,
    DifyNodeType.iteration: 5,
    DifyNodeType.parameter_extractor: 5,
    DifyNodeType.variable_aggregator: 5,
}


N8N_TYPE_TO_TARGET = {
    "@n8n/n8n-nodes-langchain.lmChatOpenAi": DifyNodeType.llm,
    "@n8n/n8n-nodes-langchain.openAi": DifyNodeType.llm,
    "@n8n/n8n-nodes-langchain.agent": DifyNodeType.llm,
    "n8n-nodes-base.code": DifyNodeType.code,
    "n8n-nodes-base.httpRequest": DifyNodeType.http_request,
    "n8n-nodes-base.set": DifyNodeType.template_transform,
    "n8n-nodes-base.googleSheets": DifyNodeType.tool,
    "n8n-nodes-base.googleDrive": DifyNodeType.tool,
    "n8n-nodes-base.gmail": DifyNodeType.tool,
    "n8n-nodes-base.slack": DifyNodeType.tool,
    "n8n-nodes-base.telegram": DifyNodeType.tool,
    "n8n-nodes-base.if": DifyNodeType.if_else,
    "n8n-nodes-base.splitInBatches": DifyNodeType.iteration,
    "@n8n/n8n-nodes-langchain.informationExtractor": DifyNodeType.parameter_extractor,
    "n8n-nodes-base.aggregate": DifyNodeType.variable_aggregator,
}


KNOWN_WORKFLOW_IDS = [
    2519,
    2896,
    2982,
    4827,
    4846,
    5035,
    5110,
    5428,
    5453,
    5683,
    5691,
    5755,
    5819,
    5906,
    5962,
    6480,
    7423,
    7639,
    7756,
    8500,
    9200,
    9437,
    9867,
    10000,
    10358,
    10427,
    11572,
    12345,
    12462,
]


def reset_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in [SAMPLES_PATH, SUMMARY_PATH, MANIFEST_PATH]:
        if path.exists():
            path.unlink()


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def build_text_signature(action_name: str, description: str) -> str:
    merged = f"{normalize_text(action_name)} || {normalize_text(description)}"
    return hashlib.sha256(merged.encode("utf-8")).hexdigest()


def request_json(url: str) -> dict:
    request = urllib.request.Request(url, headers=REQUEST_HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def safe_dump_text(value: object, limit: int = 400) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:limit]
    return json.dumps(value, ensure_ascii=False)[:limit]


def extract_from_expression(text: str) -> list[str]:
    names: list[str] = []
    if not isinstance(text, str):
        return names

    for match in [
        "url",
        "method",
        "conditions",
        "iterator_selector",
        "variables",
        "instruction",
        "template",
        "provider_id",
        "tool_name",
    ]:
        if match in text.lower():
            names.append(match)

    selectors = [
        "message",
        "messages",
        "text",
        "data",
        "items",
        "records",
        "rows",
        "url",
        "urls",
        "file",
        "files",
        "document",
        "documents",
        "result",
        "results",
        "score",
        "scores",
        "reason",
        "reasons",
        "response",
        "output",
        "caption",
        "theme",
        "photo",
        "audio",
        "video",
    ]
    lowered = text.lower()
    for token in selectors:
        if token in lowered:
            names.append(token)
    return names


def infer_inputs_from_parameters(node_type: str, parameters: dict) -> list[str]:
    inputs: list[str] = []

    for key, value in parameters.items():
        if key in {"url", "method", "text", "prompt", "template", "jsCode"}:
            inputs.append(key)
        if key == "conditions":
            inputs.append("conditions")
        if key == "aggregate":
            inputs.append("variables")

        if isinstance(value, str):
            inputs.extend(extract_from_expression(value))
        elif isinstance(value, dict):
            inputs.extend(extract_from_expression(safe_dump_text(value)))
            if key == "assignments":
                for assignment in value.get("assignments", []):
                    name = assignment.get("name")
                    if name:
                        inputs.append(str(name))
                    inputs.extend(extract_from_expression(safe_dump_text(assignment.get("value", ""))))
            if key == "attributes":
                inputs.append("instruction")
                for attribute in value.get("attributes", []):
                    name = attribute.get("name")
                    if name:
                        inputs.append(str(name))
            if key in {"bodyParameters", "headerParameters"}:
                for item in value.get("parameters", []):
                    name = item.get("name")
                    if name:
                        inputs.append(str(name))
        elif isinstance(value, list):
            for item in value:
                inputs.extend(extract_from_expression(safe_dump_text(item)))

    if node_type == "n8n-nodes-base.httpRequest":
        inputs.extend(["url", "method"])
    if node_type == "n8n-nodes-base.if":
        inputs.append("conditions")
    if node_type == "n8n-nodes-base.splitInBatches":
        inputs.append("items")
        inputs.append("iterator_selector")
    if node_type == "@n8n/n8n-nodes-langchain.informationExtractor":
        inputs.extend(["instruction", "text"])
    if node_type == "n8n-nodes-base.aggregate":
        inputs.append("variables")
    if node_type == "n8n-nodes-base.set":
        inputs.append("template")

    return unique_keep_order([item for item in inputs if item])


def infer_outputs(node_type: str, node_name: str, parameters: dict) -> list[str]:
    outputs: list[str] = []

    if node_type in {
        "@n8n/n8n-nodes-langchain.lmChatOpenAi",
        "@n8n/n8n-nodes-langchain.openAi",
        "@n8n/n8n-nodes-langchain.agent",
    }:
        outputs.append("text")
    if node_type == "n8n-nodes-base.httpRequest":
        outputs.append("response")
    if node_type == "n8n-nodes-base.aggregate":
        outputs.append("output")
    if node_type == "@n8n/n8n-nodes-langchain.informationExtractor":
        outputs.append("parameters")
    if node_type == "n8n-nodes-base.set":
        outputs.append("output")
    if node_type == "n8n-nodes-base.code":
        outputs.append("result")
    if node_type in {
        "n8n-nodes-base.googleSheets",
        "n8n-nodes-base.googleDrive",
        "n8n-nodes-base.gmail",
        "n8n-nodes-base.slack",
        "n8n-nodes-base.telegram",
    }:
        outputs.append("result")

    assignments = parameters.get("assignments", {})
    if isinstance(assignments, dict):
        for assignment in assignments.get("assignments", []):
            name = assignment.get("name")
            if name:
                outputs.append(str(name))

    if "columns" in parameters:
        outputs.append("result")

    if "aggregate" in parameters and "output" not in outputs:
        outputs.append("output")

    if "attributes" in parameters:
        outputs.append("parameters")

    lowered_name = node_name.lower()
    if "caption" in lowered_name:
        outputs.append("caption")
    if "video" in lowered_name:
        outputs.append("video")
    if "audio" in lowered_name:
        outputs.append("audio")

    return unique_keep_order([item for item in outputs if item])


def infer_available_resources(node_type: str, node_name: str) -> list[str]:
    resources: list[str] = []
    name_map = {
        "n8n-nodes-base.googleSheets": ["provider_id", "tool_name", "google_sheets"],
        "n8n-nodes-base.googleDrive": ["provider_id", "tool_name", "google_drive"],
        "n8n-nodes-base.gmail": ["provider_id", "tool_name", "gmail"],
        "n8n-nodes-base.slack": ["provider_id", "tool_name", "slack"],
        "n8n-nodes-base.telegram": ["provider_id", "tool_name", "telegram"],
    }
    resources.extend(name_map.get(node_type, []))
    if resources:
        resources.append(node_name)
    return unique_keep_order(resources)


def infer_parent_block_type(node_type: str) -> str:
    if node_type == "n8n-nodes-base.splitInBatches":
        return "Loop"
    if node_type == "n8n-nodes-base.if":
        return "Conditional"
    return "Sequential"


def infer_description(workflow_name: str, workflow_description: str, node_name: str, node_type: str, parameters: dict) -> str:
    parts = [workflow_name, node_name]
    if workflow_description:
        parts.append(str(workflow_description)[:300])

    if node_type in {
        "@n8n/n8n-nodes-langchain.lmChatOpenAi",
        "@n8n/n8n-nodes-langchain.openAi",
        "@n8n/n8n-nodes-langchain.agent",
    }:
        if "text" in parameters:
            parts.append(safe_dump_text(parameters.get("text"), 300))
        if "messages" in parameters:
            parts.append(safe_dump_text(parameters.get("messages"), 300))
        system_message = parameters.get("options", {}).get("systemMessage")
        if system_message:
            parts.append(safe_dump_text(system_message, 300))

    if node_type == "n8n-nodes-base.httpRequest":
        parts.append(safe_dump_text(parameters.get("url", ""), 180))
        parts.append(safe_dump_text(parameters.get("jsonBody", ""), 220))

    if node_type == "n8n-nodes-base.if":
        parts.append(safe_dump_text(parameters.get("conditions", ""), 260))

    if node_type == "n8n-nodes-base.code":
        parts.append(safe_dump_text(parameters.get("jsCode", ""), 260))

    if node_type == "n8n-nodes-base.set":
        parts.append(safe_dump_text(parameters.get("assignments", ""), 260))

    if node_type == "@n8n/n8n-nodes-langchain.informationExtractor":
        parts.append(safe_dump_text(parameters.get("text", ""), 240))
        parts.append(safe_dump_text(parameters.get("attributes", ""), 240))

    if node_type == "n8n-nodes-base.aggregate":
        parts.append(safe_dump_text(parameters.get("aggregate", ""), 80))

    return " ".join(part.strip() for part in parts if str(part).strip())


def infer_expected_degraded(expected_type: DifyNodeType, inputs: list[str], resources: list[str]) -> bool:
    input_set = {item.lower() for item in inputs}
    resource_set = {item.lower() for item in resources}
    if expected_type == DifyNodeType.http_request:
        return not {"url", "method"}.issubset(input_set)
    if expected_type == DifyNodeType.tool:
        return not {"provider_id", "tool_name"}.issubset(resource_set)
    if expected_type == DifyNodeType.if_else:
        return "conditions" not in input_set
    if expected_type == DifyNodeType.iteration:
        return "iterator_selector" not in input_set
    if expected_type == DifyNodeType.parameter_extractor:
        return "instruction" not in input_set
    if expected_type == DifyNodeType.variable_aggregator:
        return "variables" not in input_set
    if expected_type == DifyNodeType.template_transform:
        return "template" not in input_set
    return False


def collect_external_samples() -> tuple[list[NodeMappingEvalSample], dict]:
    collected: dict[DifyNodeType, list[NodeMappingEvalSample]] = {
        node_type: [] for node_type in TARGET_COUNTS
    }
    seen_sample_ids: set[str] = set()
    manifest: dict[str, list[int]] = {}

    for workflow_id in KNOWN_WORKFLOW_IDS:
        try:
            payload = request_json(f"{WORKFLOW_BASE_URL}/{workflow_id}")
        except (HTTPError, URLError):
            continue
        workflow_meta = payload["workflow"]
        workflow = workflow_meta["workflow"]
        workflow_name = workflow_meta["name"]
        workflow_description = workflow_meta.get("description", "")
        manifest[str(workflow_id)] = []

        for node in workflow.get("nodes", []):
            raw_type = node.get("type", "")
            if raw_type not in N8N_TYPE_TO_TARGET:
                continue

            expected_type = N8N_TYPE_TO_TARGET[raw_type]
            if len(collected[expected_type]) >= TARGET_COUNTS[expected_type]:
                continue

            node_id = str(node.get("id", ""))
            sample_id = f"n8n::{workflow_id}::{node_id}"
            if sample_id in seen_sample_ids:
                continue

            node_name = str(node.get("name", raw_type))
            parameters = node.get("parameters", {})
            inputs = infer_inputs_from_parameters(raw_type, parameters)
            outputs = infer_outputs(raw_type, node_name, parameters)
            resources = infer_available_resources(raw_type, node_name)
            description = infer_description(
                workflow_name=workflow_name,
                workflow_description=workflow_description,
                node_name=node_name,
                node_type=raw_type,
                parameters=parameters,
            )
            sample = NodeMappingEvalSample(
                sample_id=sample_id,
                workflow_id=f"n8n::{workflow_id}",
                source_node_id=node_id,
                source="external_n8n",
                split="external",
                expected_node_type=expected_type,
                action_name=node_name,
                description=description,
                inputs=inputs,
                outputs=outputs,
                available_resources=resources,
                parent_block_type=infer_parent_block_type(raw_type),
                difficulty="external",
                expected_degraded=infer_expected_degraded(expected_type, inputs, resources),
                tags=["n8n", raw_type, workflow_name],
                text_signature=build_text_signature(node_name, description),
            )
            collected[expected_type].append(sample)
            seen_sample_ids.add(sample_id)
            manifest[str(workflow_id)].append(node_id)

        time.sleep(0.15)

        if all(len(collected[node_type]) >= target for node_type, target in TARGET_COUNTS.items()):
            break

    missing = {
        node_type.value: TARGET_COUNTS[node_type] - len(collected[node_type])
        for node_type in TARGET_COUNTS
        if len(collected[node_type]) < TARGET_COUNTS[node_type]
    }
    if missing:
        raise RuntimeError(f"External sample collection is incomplete: {missing}")

    samples: list[NodeMappingEvalSample] = []
    for node_type, items in collected.items():
        samples.extend(items[: TARGET_COUNTS[node_type]])

    return samples, manifest


def build_summary(samples: list[NodeMappingEvalSample], manifest: dict[str, list[int]]) -> dict:
    distribution = Counter(sample.expected_node_type.value for sample in samples)
    workflows = sorted({sample.workflow_id for sample in samples})
    return {
        "sample_count": len(samples),
        "workflow_count": len(workflows),
        "distribution": dict(sorted(distribution.items())),
        "workflows": workflows,
        "manifest": manifest,
    }


def main() -> None:
    reset_outputs()
    samples, manifest = collect_external_samples()
    for sample in samples:
        append_to_jsonl(str(SAMPLES_PATH), sample)

    summary = build_summary(samples, manifest)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
