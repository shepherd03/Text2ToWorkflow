import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.core.schema import NodeMappingEvalSample


DATASET_PATH = Path("generated_data/dify_external_dataset/dataset.jsonl")
SOURCE_SPLIT_DIR = Path("generated_data/dsl_generation/dify_external_node_mapping_eval_from_dataset")
TARGET_EVAL_DIR = Path("generated_data/dsl_generation/dify_external_node_mapping_eval")
FULL_EXTERNAL_SOURCE_NAMES = [
    "train_samples.jsonl",
    "valid_samples.jsonl",
    "test_samples.jsonl",
]


def copy_split(source_name: str, target_name: str) -> int:
    source_path = SOURCE_SPLIT_DIR / source_name
    target_path = TARGET_EVAL_DIR / target_name
    if not source_path.exists():
        raise FileNotFoundError(f"Missing split file: {source_path}")

    TARGET_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path.unlink()

    count = 0
    with source_path.open("r", encoding="utf-8") as src, target_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            sample = NodeMappingEvalSample.model_validate(json.loads(line))
            sample = sample.model_copy(update={"source": "external_dify_dataset", "split": "external"})
            dst.write(json.dumps(sample.model_dump(), ensure_ascii=False) + "\n")
            count += 1
    return count


def build_full_external_pool(target_name: str = "external_samples.jsonl") -> int:
    target_path = TARGET_EVAL_DIR / target_name
    TARGET_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path.unlink()

    seen_sample_ids: set[str] = set()
    count = 0
    with target_path.open("w", encoding="utf-8") as dst:
        for source_name in FULL_EXTERNAL_SOURCE_NAMES:
            source_path = SOURCE_SPLIT_DIR / source_name
            if not source_path.exists():
                raise FileNotFoundError(f"Missing split file: {source_path}")
            with source_path.open("r", encoding="utf-8") as src:
                for line in src:
                    if not line.strip():
                        continue
                    sample = NodeMappingEvalSample.model_validate(json.loads(line))
                    if sample.sample_id in seen_sample_ids:
                        continue
                    seen_sample_ids.add(sample.sample_id)
                    sample = sample.model_copy(
                        update={"source": "external_dify_dataset", "split": "external_full"}
                    )
                    dst.write(json.dumps(sample.model_dump(), ensure_ascii=False) + "\n")
                    count += 1
    return count


def main() -> None:
    for stale_name in [
        "train_samples.jsonl",
        "valid_samples.jsonl",
        "external_samples.jsonl",
        "external_predictions.jsonl",
        "external_evaluation_summary.json",
        "external_summary.json",
    ]:
        stale_path = TARGET_EVAL_DIR / stale_name
        if stale_path.exists():
            stale_path.unlink()

    counts = {
        "train": copy_split("train_samples.jsonl", "train_samples.jsonl"),
        "valid": copy_split("valid_samples.jsonl", "valid_samples.jsonl"),
        "test": copy_split("test_samples.jsonl", "test_samples.jsonl"),
    }
    full_external_count = build_full_external_pool("external_samples.jsonl")

    summary = {
        "dataset_path": str(DATASET_PATH),
        "source_split_dir": str(SOURCE_SPLIT_DIR),
        "target_eval_dir": str(TARGET_EVAL_DIR),
        "counts": counts,
        "external_eval_sample_count": full_external_count,
        "external_eval_mode": "full_external_pool",
    }

    (TARGET_EVAL_DIR / "external_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
