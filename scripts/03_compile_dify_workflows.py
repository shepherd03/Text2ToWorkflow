from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.schema import SequentialBlock, UTR
from src.dsl_generation.pipeline import DSLGenerationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile UTR and Skeleton records into Dify workflow DSL.")
    parser.add_argument(
        "--utr-file",
        default="generated_data/utr_generation/utrs.jsonl",
        help="Input UTR JSONL file.",
    )
    parser.add_argument(
        "--skeleton-file",
        default="generated_data/skeleton_planning/iter2/skeletons.jsonl",
        help="Input Skeleton JSONL file.",
    )
    parser.add_argument(
        "--output-file",
        default="generated_data/dsl_generation/dsls.jsonl",
        help="Output JSONL file for compiled workflows.",
    )
    parser.add_argument(
        "--error-file",
        default="generated_data/dsl_generation/errors.json",
        help="Output JSON file for compile errors.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Optional maximum number of records to process. 0 means all records.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def index_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record.get("id", "")): record for record in records if record.get("id")}


def main() -> None:
    args = parse_args()
    utr_file = Path(args.utr_file)
    skeleton_file = Path(args.skeleton_file)
    output_file = Path(args.output_file)
    error_file = Path(args.error_file)

    if not utr_file.exists():
        raise FileNotFoundError(f"UTR file not found: {utr_file}")
    if not skeleton_file.exists():
        raise FileNotFoundError(f"Skeleton file not found: {skeleton_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    error_file.parent.mkdir(parents=True, exist_ok=True)

    utr_records = index_by_id(load_jsonl(utr_file))
    skeleton_records = load_jsonl(skeleton_file)
    if args.max_records > 0:
        skeleton_records = skeleton_records[: args.max_records]

    pipeline = DSLGenerationPipeline()
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for skeleton_record in skeleton_records:
        record_id = str(skeleton_record.get("id", ""))
        instruction = str(skeleton_record.get("instruction", ""))
        try:
            utr_record = utr_records.get(record_id)
            if utr_record is None:
                raise KeyError(f"Matching UTR record not found for id={record_id}")

            utr = UTR.model_validate(utr_record["utr"])
            skeleton = SequentialBlock.model_validate(skeleton_record["skeleton"])
            output = pipeline.run_step3_minimal(utr, skeleton)
            if not output.compile_report.success:
                raise RuntimeError("; ".join(output.compile_report.errors) or "DSL compile failed")

            results.append(
                {
                    "id": record_id,
                    "instruction": instruction or str(utr_record.get("instruction", "")),
                    "workflow": output.workflow.model_dump() if output.workflow else None,
                    "compiled_graph": output.compiled_graph.model_dump() if output.compiled_graph else None,
                    "node_mappings": [item.model_dump() for item in output.node_mappings],
                    "compile_report": output.compile_report.model_dump(),
                    "normalization_report": output.normalization_output.normalization_report.model_dump(),
                    "precheck_report": output.normalization_output.precheck_report.model_dump(),
                }
            )
            print(f"[OK] compiled workflow: {record_id}")
        except Exception as exc:
            errors.append({"id": record_id, "instruction": instruction, "error": str(exc)})
            print(f"[ERROR] failed workflow compile: {record_id}: {exc}")

    with output_file.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")

    error_file.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Completed. Success: {len(results)}, Errors: {len(errors)}")
    print(f"Results: {output_file}")
    print(f"Errors: {error_file}")


if __name__ == "__main__":
    main()
