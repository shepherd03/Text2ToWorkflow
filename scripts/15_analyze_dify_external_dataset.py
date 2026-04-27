import json
import os
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.dsl_generation.eval_sample_utils import ALLOWED_DIFY_NODE_TYPES


DATASET_PATH = Path("generated_data/dify_external_dataset/dataset.jsonl")
SUMMARY_PATH = Path("generated_data/dify_external_dataset/analysis.json")


def main() -> None:
    workflow_count = 0
    source_counter: Counter[str] = Counter()
    node_counter: Counter[str] = Counter()
    instruction_lengths: list[int] = []

    for line in DATASET_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        workflow_count += 1
        item = json.loads(line)
        metadata = item.get("metadata", {})
        source_counter.update([metadata.get("source", "unknown")])
        instruction_lengths.append(len(str(item.get("instruction", ""))))

        dsl = yaml.safe_load(item["dsl"])
        for node in dsl.get("workflow", {}).get("graph", {}).get("nodes", []):
            node_data = node.get("data", {})
            raw_type = node_data.get("type")
            if raw_type not in ALLOWED_DIFY_NODE_TYPES:
                continue
            node_counter.update([ALLOWED_DIFY_NODE_TYPES[raw_type].value])

    summary = {
        "workflow_count": workflow_count,
        "source_distribution": dict(sorted(source_counter.items())),
        "node_distribution": dict(sorted(node_counter.items())),
        "total_supported_nodes": sum(node_counter.values()),
        "instruction_length": {
            "min": min(instruction_lengths) if instruction_lengths else 0,
            "max": max(instruction_lengths) if instruction_lengths else 0,
            "avg": (
                sum(instruction_lengths) / len(instruction_lengths)
                if instruction_lengths
                else 0.0
            ),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
