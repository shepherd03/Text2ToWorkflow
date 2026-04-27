from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from src.core.schema import DifyNodeType, NodeMappingEvalSample
from src.core.utils import unique_keep_order


ALLOWED_DIFY_NODE_TYPES = {
    "llm": DifyNodeType.llm,
    "code": DifyNodeType.code,
    "http-request": DifyNodeType.http_request,
    "template-transform": DifyNodeType.template_transform,
    "tool": DifyNodeType.tool,
    "if-else": DifyNodeType.if_else,
    "iteration": DifyNodeType.iteration,
    "parameter-extractor": DifyNodeType.parameter_extractor,
    "variable-aggregator": DifyNodeType.variable_aggregator,
}


GENERIC_TITLE_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "llm": [
        re.compile(
            r"^\s*(llm|chat\s*model|reasoning\s*model|openai\s*chat\s*model|model)\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(gpt[- ]?4o|deepseek(?:-reasoner)?|claude(?:\s*chat\s*model)?|gemini)\s*$",
            re.IGNORECASE,
        ),
    ],
    "code": [
        re.compile(r"^\s*(code|script|代码|代码执行|脚本)\s*$", re.IGNORECASE),
    ],
    "http-request": [
        re.compile(
            r"^\s*(http\s*request|request|api\s*request|接口请求|网络请求)\s*$",
            re.IGNORECASE,
        ),
    ],
    "template-transform": [
        re.compile(
            r"^\s*(template\s*transform|template|assigner|模板转换|模板处理)\s*$",
            re.IGNORECASE,
        ),
    ],
    "tool": [
        re.compile(r"^\s*(tool|工具)\s*$", re.IGNORECASE),
    ],
    "if-else": [
        re.compile(r"^\s*(if\s*/?\s*else|if|条件分支)\s*$", re.IGNORECASE),
    ],
    "iteration": [
        re.compile(r"^\s*(iteration|loop|for each|循环)\s*$", re.IGNORECASE),
    ],
    "parameter-extractor": [
        re.compile(
            r"^\s*(parameter\s*extractor|information\s*extractor|参数提取器)\s*$",
            re.IGNORECASE,
        ),
    ],
    "variable-aggregator": [
        re.compile(
            r"^\s*(variable\s*aggregator|aggregator|变量聚合器)\s*$",
            re.IGNORECASE,
        ),
    ],
}


GENERIC_PREFIX_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*llm\s*[-:：]\s*", re.IGNORECASE),
    re.compile(r"^\s*code\s*[-:：]\s*", re.IGNORECASE),
    re.compile(r"^\s*http\s*request\s*[-:：]\s*", re.IGNORECASE),
    re.compile(r"^\s*request\s*[-:：]\s*", re.IGNORECASE),
    re.compile(r"^\s*tool\s*[-:：]\s*", re.IGNORECASE),
    re.compile(r"^\s*iteration\s*[-:：]\s*", re.IGNORECASE),
    re.compile(r"^\s*template\s*transform\s*[-:：]\s*", re.IGNORECASE),
    re.compile(r"^\s*parameter\s*extractor\s*[-:：]\s*", re.IGNORECASE),
    re.compile(r"^\s*information\s*extractor\s*[-:：]\s*", re.IGNORECASE),
]


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def build_text_signature(action_name: str, description: str) -> str:
    merged = f"{normalize_text(action_name)} || {normalize_text(description)}"
    return hashlib.sha256(merged.encode("utf-8")).hexdigest()


def safe_dump_text(value: object, limit: int = 400) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:limit]
    return json.dumps(value, ensure_ascii=False)[:limit]


def canonicalize_dsl_text(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    # Some GitHub issue repros use "app:/" instead of valid YAML root "app:".
    return re.sub(r"^app:/\s*$", "app:", text, flags=re.MULTILINE)


def sanitize_action_name(raw_title: str, raw_type: str, mode: str = "raw") -> str:
    title = str(raw_title or "").strip()
    if mode == "raw":
        return title
    if mode == "blank":
        return ""

    cleaned = title
    for pattern in GENERIC_PREFIX_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = cleaned.strip(" -:：")

    for pattern in GENERIC_TITLE_PATTERNS.get(raw_type, []):
        if pattern.match(cleaned or title):
            return ""

    if cleaned.lower() == raw_type:
        return ""
    return cleaned


def _selector_tail_name(value: Any) -> str:
    if isinstance(value, (list, tuple)) and value:
        return str(value[-1])
    return ""


def _collect_tool_parameter_names(tool_parameters: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for key, config in tool_parameters.items():
        names.append(str(key))
        if isinstance(config, dict):
            nested_name = _selector_tail_name(config.get("value"))
            if nested_name:
                names.append(nested_name)
    return names


def infer_inputs_from_dify_node(node_data: dict[str, Any]) -> list[str]:
    collected: list[str] = []

    for variable in node_data.get("variables", []):
        if isinstance(variable, dict):
            name = variable.get("variable") or variable.get("label")
            if name:
                collected.append(str(name))
            selector_name = _selector_tail_name(variable.get("value_selector"))
            if selector_name:
                collected.append(selector_name)
        elif isinstance(variable, (list, tuple)) and variable:
            collected.append(str(variable[-1]))

    for selector in [
        node_data.get("iterator_selector"),
        node_data.get("output_selector"),
        node_data.get("query"),
    ]:
        selector_name = _selector_tail_name(selector)
        if selector_name:
            collected.append(selector_name)

    for key in ["url", "method", "instruction", "query", "template"]:
        if key in node_data:
            collected.append(key)

    if node_data.get("conditions") or node_data.get("cases"):
        collected.append("conditions")
    if "iterator_selector" in node_data:
        collected.append("iterator_selector")
    if node_data.get("type") == "variable-aggregator" and node_data.get("variables"):
        collected.append("variables")
    if node_data.get("tool_name"):
        collected.append("tool_name")
    if node_data.get("provider_id"):
        collected.append("provider_id")
    if isinstance(node_data.get("tool_parameters"), dict):
        collected.extend(_collect_tool_parameter_names(node_data["tool_parameters"]))

    for parameter in node_data.get("parameters", []):
        if isinstance(parameter, dict):
            name = parameter.get("name")
            if name:
                collected.append(str(name))

    body = node_data.get("body")
    if isinstance(body, dict):
        collected.append("body")
        data_selector = _selector_tail_name(body.get("data"))
        if data_selector:
            collected.append(data_selector)

    return unique_keep_order([item for item in collected if item])


def infer_outputs_from_dify_node(node_data: dict[str, Any]) -> list[str]:
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

    node_type = node_data.get("type")
    if node_type == "llm":
        collected.append("text")
    if node_type == "tool":
        collected.append("result")
    if node_type == "http-request":
        collected.append("response")
    if node_type == "parameter-extractor":
        collected.append("parameters")
    if node_type == "variable-aggregator":
        collected.append("output")
    if node_type == "template-transform":
        collected.append("output")

    return unique_keep_order([item for item in collected if item])


def infer_available_resources_from_dify_node(node_data: dict[str, Any]) -> list[str]:
    resources: list[str] = []
    for key in ["provider_name", "provider_id", "tool_name", "tool_label"]:
        value = node_data.get(key)
        if value:
            resources.append(str(value))
    return unique_keep_order(resources)


def infer_description_from_dify_node(
    node_data: dict[str, Any],
    *,
    include_title_as_fallback: bool = True,
) -> str:
    parts: list[str] = []

    for key in ["desc", "instruction", "query"]:
        value = node_data.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)

    prompt_template = node_data.get("prompt_template", [])
    if isinstance(prompt_template, list):
        for item in prompt_template[:3]:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"])[:400])

    template = node_data.get("template")
    if template:
        parts.append(safe_dump_text(template, 260))

    code = node_data.get("code")
    if code:
        parts.append(safe_dump_text(code, 260))

    if node_data.get("type") == "http-request":
        parts.append(safe_dump_text(node_data.get("url", ""), 180))
        parts.append(safe_dump_text(node_data.get("body", ""), 240))
        parts.append(safe_dump_text(node_data.get("headers", ""), 120))

    if node_data.get("type") == "parameter-extractor":
        parts.append(safe_dump_text(node_data.get("parameters", ""), 260))

    if node_data.get("type") == "if-else":
        parts.append(safe_dump_text(node_data.get("cases", ""), 260))
        parts.append(safe_dump_text(node_data.get("conditions", ""), 260))

    if node_data.get("type") == "tool":
        parts.append(safe_dump_text(node_data.get("tool_configurations", ""), 220))

    if include_title_as_fallback and not parts and node_data.get("title"):
        parts.append(str(node_data["title"]))

    return " ".join(part.strip() for part in parts if str(part).strip())


def infer_parent_block_type_from_dify_node(node_data: dict[str, Any]) -> str:
    if node_data.get("isInIteration"):
        return "Loop"
    if node_data.get("type") == "if-else":
        return "Conditional"
    if node_data.get("type") == "iteration":
        return "Loop"
    return "Sequential"


def infer_expected_degraded(
    expected_type: DifyNodeType,
    inputs: list[str],
    resources: list[str],
) -> bool:
    input_set = {item.lower() for item in inputs}
    resource_set = {item.lower() for item in resources}
    if expected_type == DifyNodeType.http_request:
        return not {"url", "method"}.issubset(input_set)
    if expected_type == DifyNodeType.tool:
        return not (
            {"provider_id", "tool_name"}.issubset(resource_set)
            or {"provider_id", "tool_name"}.issubset(input_set)
        )
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


def build_node_overlap_key(node_data: dict[str, Any]) -> str:
    payload = {
        "type": node_data.get("type", ""),
        "title": node_data.get("title", ""),
        "description": infer_description_from_dify_node(node_data, include_title_as_fallback=True),
        "tool_name": node_data.get("tool_name", ""),
        "provider_id": node_data.get("provider_id", ""),
        "url": node_data.get("url", ""),
        "method": node_data.get("method", ""),
        "template": safe_dump_text(node_data.get("template", ""), 180),
        "code": safe_dump_text(node_data.get("code", ""), 180),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_eval_sample_from_dify_node(
    *,
    workflow_id: str,
    source_node_id: str,
    node_data: dict[str, Any],
    source: str,
    split: str,
    difficulty: str,
    tags: list[str] | None = None,
    title_mode: str = "raw",
) -> NodeMappingEvalSample | None:
    raw_type = node_data.get("type")
    if raw_type not in ALLOWED_DIFY_NODE_TYPES:
        return None

    mapped_type = ALLOWED_DIFY_NODE_TYPES[raw_type]
    action_name = sanitize_action_name(
        str(node_data.get("title") or raw_type),
        raw_type,
        mode=title_mode,
    )
    description = infer_description_from_dify_node(
        node_data,
        include_title_as_fallback=title_mode == "raw",
    )
    inputs = infer_inputs_from_dify_node(node_data)
    outputs = infer_outputs_from_dify_node(node_data)
    resources = infer_available_resources_from_dify_node(node_data)
    parent_block_type = infer_parent_block_type_from_dify_node(node_data)

    return NodeMappingEvalSample(
        sample_id=f"{workflow_id}::{source_node_id}",
        workflow_id=workflow_id,
        source_node_id=source_node_id,
        source=source,
        split=split,
        expected_node_type=mapped_type,
        action_name=action_name,
        description=description,
        inputs=inputs,
        outputs=outputs,
        available_resources=resources,
        parent_block_type=parent_block_type,
        difficulty=difficulty,
        expected_degraded=infer_expected_degraded(mapped_type, inputs, resources),
        tags=tags or [raw_type],
        text_signature=build_text_signature(action_name, description),
    )
