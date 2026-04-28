from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_settings
from src.core.llm_client import DeepSeekClient
from src.core.utils import append_to_jsonl
from src.dsl_generation.dify_external_dataset_utils import (
    build_workflow_overlap_keys,
    load_dataset_overlap_keys,
)

SOURCE_DATASET_PATH = ROOT / "generated_data/dify_external_dataset/dataset.jsonl"
OUTPUT_DIR = ROOT / "generated_data/dify_external_dataset_research/latest"


def load_external_builder():
    path = ROOT / "scripts/13_build_dify_external_dataset.py"
    spec = importlib.util.spec_from_file_location("external_dataset_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small auditable research dataset from existing external DSLs with real LLM instructions."
    )
    parser.add_argument("--source-file", default=str(SOURCE_DATASET_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR.relative_to(ROOT)))
    parser.add_argument("--max-records", type=int, default=8)
    parser.add_argument("--min-novel-ratio", type=float, default=0.10)
    return parser.parse_args(argv)


def load_source_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = load_settings()
    if not settings.llm_enabled:
        raise RuntimeError("DEEPSEEK_API_KEY 未配置，不能构造真实 LLM instruction 数据。")

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dataset.jsonl"
    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"
    for path in [dataset_path, manifest_path, summary_path]:
        if path.exists():
            path.unlink()

    external_builder = load_external_builder()
    client = DeepSeekClient(settings)
    source_records = load_source_records(Path(args.source_file))
    baseline_keys = load_dataset_overlap_keys(ROOT / "dataset/dataset.jsonl")

    accepted = 0
    skipped = 0
    errors: list[dict[str, Any]] = []
    selected_sources: list[dict[str, Any]] = []

    for source_record in source_records:
        if accepted >= args.max_records:
            break
        dsl_text = str(source_record.get("dsl", ""))
        overlap_keys = build_workflow_overlap_keys(dsl_text)
        if not overlap_keys:
            skipped += 1
            continue
        novel_ratio = len(overlap_keys - baseline_keys) / len(overlap_keys)
        if novel_ratio < args.min_novel_ratio:
            skipped += 1
            continue

        source_url = str(source_record.get("metadata", {}).get("source_url", "existing_external_dataset"))
        record_id = f"research_ext_{accepted + 1:04d}"
        try:
            instruction = external_builder.generate_instruction(client, dsl_text, source_url)
            record = external_builder.build_dataset_record(
                record_id=record_id,
                instruction=instruction,
                dsl_text=dsl_text,
                source_type="existing_external_research",
                source_url=source_url,
            )
            record["metadata"]["source_record_id"] = source_record.get("id", "")
            record["metadata"]["novel_ratio"] = novel_ratio
            append_to_jsonl(str(dataset_path), record)
            accepted += 1
            selected_sources.append(
                {
                    "id": record_id,
                    "source_record_id": source_record.get("id", ""),
                    "source_url": source_url,
                    "novel_ratio": novel_ratio,
                    "node_count": record["metadata"]["node_count"],
                    "supported_node_type_counts": record["metadata"]["supported_node_type_counts"],
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "source_record_id": source_record.get("id", ""),
                    "source_url": source_url,
                    "error": str(exc),
                }
            )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(Path(args.source_file)),
        "output_dataset_path": str(dataset_path.relative_to(ROOT)),
        "workflow_count": accepted,
        "skipped_count": skipped,
        "error_count": len(errors),
        "llm_instruction_generation": {
            "enabled": True,
            "used_real_llm": client.call_count > 0,
            "llm_call_count": client.call_count,
            "model": settings.deepseek_model,
            "base_url": settings.deepseek_base_url,
        },
    }
    manifest = {
        **summary,
        "selected_sources": selected_sources,
        "errors": errors,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
