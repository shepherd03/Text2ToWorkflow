import importlib.util
import json

import yaml

from src.core.schema import DifyNodeType
from src.dsl_generation.eval_sample_utils import (
    build_eval_sample_from_dify_node,
    build_node_overlap_key,
    canonicalize_dsl_text,
    sanitize_action_name,
)


def _load_module(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prepare_module = _load_module(
    "prepare_dify_external_node_mapping_eval_data",
    "scripts/11_prepare_dify_external_node_mapping_eval_data.py",
)
eval_module = _load_module(
    "evaluate_dify_external_node_mapping",
    "scripts/12_evaluate_dify_external_node_mapping.py",
)


def test_canonicalize_dsl_text_repairs_issue_style_root():
    raw = "app:/\n  description: ''\nkind: app\nworkflow:\n  graph:\n    nodes: []\n"
    fixed = canonicalize_dsl_text(raw)

    assert fixed.startswith("app:\n")
    parsed = yaml.safe_load(fixed)
    assert parsed["kind"] == "app"


def test_sanitize_action_name_drops_generic_llm_title():
    assert sanitize_action_name("LLM", "llm", mode="sanitized") == ""
    assert sanitize_action_name("Code - Parse JSON", "code", mode="sanitized") == "Parse JSON"


def test_build_eval_sample_from_dify_tool_node():
    node_data = {
        "type": "tool",
        "title": "TavilySearch",
        "provider_id": "tavily",
        "provider_name": "tavily",
        "tool_name": "tavily_search",
        "tool_label": "TavilySearch",
        "tool_parameters": {
            "query": {"type": "mixed", "value": "{{#start.query#}}"},
        },
    }

    sample = build_eval_sample_from_dify_node(
        workflow_id="wf_x",
        source_node_id="node_1",
        node_data=node_data,
        source="external",
        split="external",
        difficulty="external",
        tags=["tool"],
        title_mode="sanitized",
    )

    assert sample is not None
    assert sample.expected_node_type == DifyNodeType.tool
    assert "query" in sample.inputs
    assert "tavily" in [item.lower() for item in sample.available_resources]


def test_build_node_overlap_key_changes_with_distinct_semantics():
    left = {
        "type": "http-request",
        "title": "Request A",
        "url": "https://api.a.com",
        "method": "post",
    }
    right = {
        "type": "http-request",
        "title": "Request B",
        "url": "https://api.b.com",
        "method": "post",
    }

    assert build_node_overlap_key(left) != build_node_overlap_key(right)


def test_issue_dsl_block_regex_can_decode_clipboard_payload():
    payload = (
        "data-snippet-clipboard-copy-content=\\\"app:/\\\\n  description: ''\\\\n"
        "kind: app\\\\nworkflow:\\\\n  graph:\\\\n    nodes: []\\\\n\\\""
    )
    blocks = prepare_module.extract_dsl_blocks_from_issue_html(payload)

    assert len(blocks) == 1
    assert blocks[0].startswith("app:\n")


def test_sanitize_external_sample_removes_trivial_llm_leakage():
    sample = build_eval_sample_from_dify_node(
        workflow_id="wf_x",
        source_node_id="node_2",
        node_data={
            "type": "llm",
            "title": "LLM",
            "prompt_template": [{"role": "system", "text": "Summarize the article"}],
        },
        source="external",
        split="external",
        difficulty="external",
        tags=["llm"],
        title_mode="sanitized",
    )

    cleaned = prepare_module.sanitize_external_sample(sample)

    assert cleaned.action_name == ""
    assert cleaned.outputs == []


def test_dify_external_eval_confusion_matrix():
    prediction = eval_module.NodeMappingEvalPrediction(
        sample_id="sample",
        expected_node_type=DifyNodeType.llm,
        predicted_node_type=DifyNodeType.code,
        correct=False,
    )
    matrix = eval_module.build_confusion_matrix([prediction])

    assert matrix["llm"]["code"] == 1


def test_summary_payload_shape_is_json_serializable():
    payload = {
        "backend": "tfidf",
        "sample_count": 3,
        "accuracy": 0.5,
        "macro_f1": 0.4,
    }
    encoded = json.dumps(payload, ensure_ascii=False)
    assert "tfidf" in encoded


def test_build_full_external_pool_merges_multiple_splits(tmp_path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    source_dir.mkdir()

    sample_train = {
        "sample_id": "wf::1",
        "workflow_id": "wf",
        "source_node_id": "1",
        "source": "dataset",
        "split": "train",
        "expected_node_type": "llm",
        "action_name": "ask",
        "description": "answer the user",
        "inputs": [],
        "outputs": ["text"],
        "available_resources": [],
        "parent_block_type": "Sequential",
        "difficulty": "standard",
        "expected_degraded": False,
        "tags": ["llm"],
        "text_signature": "sig1",
    }
    sample_test = {
        "sample_id": "wf::2",
        "workflow_id": "wf",
        "source_node_id": "2",
        "source": "dataset",
        "split": "test",
        "expected_node_type": "code",
        "action_name": "run code",
        "description": "import json",
        "inputs": ["query"],
        "outputs": ["result"],
        "available_resources": [],
        "parent_block_type": "Sequential",
        "difficulty": "standard",
        "expected_degraded": False,
        "tags": ["code"],
        "text_signature": "sig2",
    }
    for name, rows in {
        "train_samples.jsonl": [sample_train],
        "valid_samples.jsonl": [],
        "test_samples.jsonl": [sample_test],
    }.items():
        path = source_dir / name
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )

    prepare_eval_module = _load_module(
        "prepare_dify_external_eval_from_dataset",
        "scripts/14_prepare_dify_external_eval_from_dataset.py",
    )
    prepare_eval_module.SOURCE_SPLIT_DIR = source_dir
    prepare_eval_module.TARGET_EVAL_DIR = target_dir

    count = prepare_eval_module.build_full_external_pool()

    assert count == 2
    merged_lines = [
        json.loads(line)
        for line in (target_dir / "external_samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(merged_lines) == 2
    assert {item["sample_id"] for item in merged_lines} == {"wf::1", "wf::2"}
    assert all(item["split"] == "external_full" for item in merged_lines)
