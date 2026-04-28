from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_THRESHOLDS: dict[str, int | float] = {
    "min_utr_records": 1,
    "min_skeleton_records": 1,
    "min_external_dify_workflows": 1,
    "min_external_supported_nodes": 1,
    "min_internal_test_accuracy": 0.90,
    "min_external_accuracy": 0.95,
    "max_internal_confidence_ece": 0.25,
    "max_external_confidence_ece": 0.25,
    "max_smoke_errors": 0,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or collect project health baselines.")
    parser.add_argument(
        "--output-file",
        default="generated_data/project_health/latest.json",
        help="Path to write the healthcheck JSON summary.",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run python -m pytest tests/ and record the result.",
    )
    parser.add_argument(
        "--run-smoke",
        action="store_true",
        help="Run Stage 3 DSL smoke compile and record the result.",
    )
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> int | None:
    if not path.exists():
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "success": completed.returncode == 0,
        "stdout_tail": completed.stdout.splitlines()[-20:],
        "stderr_tail": completed.stderr.splitlines()[-20:],
    }


def parse_pytest_summary(lines: list[str]) -> dict[str, Any]:
    text = "\n".join(lines)
    passed_match = re.search(r"(\d+)\s+passed", text)
    failed_match = re.search(r"(\d+)\s+failed", text)
    skipped_match = re.search(r"(\d+)\s+skipped", text)
    error_match = re.search(r"(\d+)\s+errors?", text)
    return {
        "passed": int(passed_match.group(1)) if passed_match else 0,
        "failed": int(failed_match.group(1)) if failed_match else 0,
        "skipped": int(skipped_match.group(1)) if skipped_match else 0,
        "errors": int(error_match.group(1)) if error_match else 0,
    }


def parse_stage3_smoke_summary(lines: list[str]) -> dict[str, Any]:
    text = "\n".join(lines)
    match = re.search(r"Success:\s*(\d+),\s*Errors:\s*(\d+)", text)
    if not match:
        return {"compiled_records": None, "error_count": None}
    return {
        "compiled_records": int(match.group(1)),
        "error_count": int(match.group(2)),
    }


def collect_artifacts() -> dict[str, Any]:
    paths = {
        "utr_records": ROOT / "generated_data/utr_generation/utrs.jsonl",
        "skeleton_records": ROOT / "generated_data/skeleton_planning/iter2/skeletons.jsonl",
        "dsl_records": ROOT / "generated_data/dsl_generation/dsls.jsonl",
        "external_dify_dataset": ROOT / "generated_data/dify_external_dataset/dataset.jsonl",
        "node_mapping_train": ROOT / "generated_data/dsl_generation/node_mapping_eval/train_samples.jsonl",
        "node_mapping_valid": ROOT / "generated_data/dsl_generation/node_mapping_eval/valid_samples.jsonl",
        "node_mapping_test": ROOT / "generated_data/dsl_generation/node_mapping_eval/test_samples.jsonl",
        "llm_research_records": ROOT / "generated_data/llm_workflow_research/latest/records.jsonl",
    }
    return {
        name: {
            "path": str(path.relative_to(ROOT)),
            "exists": path.exists(),
            "records": count_jsonl(path),
        }
        for name, path in paths.items()
    }


def extract_node_mapping_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"available": False}
    result: dict[str, Any] = {"available": True, "backend": payload.get("backend")}
    for split in ["valid", "test", "hard"]:
        metrics = payload.get(split, {}).get("metrics", {})
        if metrics:
            result[split] = {
                "sample_count": metrics.get("sample_count"),
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "degradation_accuracy": metrics.get("degradation_accuracy"),
                "degradation_detection_accuracy": metrics.get(
                    "degradation_detection_accuracy"
                ),
                "degradation_type_accuracy": metrics.get("degradation_type_accuracy"),
                "confidence_ece": metrics.get("confidence_ece"),
                "confidence_brier": metrics.get("confidence_brier"),
                "seen_accuracy": metrics.get("seen_accuracy"),
                "unseen_accuracy": metrics.get("unseen_accuracy"),
            }
    return result


def collect_metrics() -> dict[str, Any]:
    internal_summary = read_json(
        ROOT / "generated_data/dsl_generation/node_mapping_eval/evaluation_summary.json"
    )
    external_summary = read_json(
        ROOT / "generated_data/dsl_generation/dify_external_node_mapping_eval/external_evaluation_summary.json"
    )
    external_dataset = read_json(ROOT / "generated_data/dify_external_dataset/analysis.json")
    llm_research = read_json(ROOT / "generated_data/llm_workflow_research/latest/manifest.json")
    return {
        "node_mapping_internal": extract_node_mapping_summary(
            internal_summary if isinstance(internal_summary, dict) else None
        ),
        "node_mapping_external": external_summary if isinstance(external_summary, dict) else {"available": False},
        "external_dify_dataset": external_dataset if isinstance(external_dataset, dict) else {"available": False},
        "llm_workflow_research": llm_research if isinstance(llm_research, dict) else {"available": False},
    }


def run_stage3_smoke() -> dict[str, Any]:
    temp_dir = Path(tempfile.gettempdir())
    output_file = temp_dir / "utr_healthcheck_dsls.jsonl"
    error_file = temp_dir / "utr_healthcheck_errors.json"
    result = run_command(
        [
            sys.executable,
            "scripts/03_compile_dify_workflows.py",
            "--output-file",
            str(output_file),
            "--error-file",
            str(error_file),
        ]
    )
    errors = read_json(error_file)
    parsed = parse_stage3_smoke_summary(result["stdout_tail"])
    result["compiled_records"] = count_jsonl(output_file)
    if result["compiled_records"] is None:
        result["compiled_records"] = parsed["compiled_records"]
    result["error_count"] = len(errors) if isinstance(errors, list) else parsed["error_count"]
    return result


def _metric_value(payload: dict[str, Any], path: list[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def build_quality_gates(payload: dict[str, Any]) -> dict[str, Any]:
    thresholds = dict(DEFAULT_THRESHOLDS)
    checks: list[dict[str, Any]] = []

    def add_check(name: str, actual: Any, expected: Any, passed: bool, severity: str = "error") -> None:
        checks.append(
            {
                "name": name,
                "actual": actual,
                "expected": expected,
                "passed": passed,
                "severity": severity,
            }
        )

    artifacts = payload.get("artifacts", {})
    add_check(
        "utr_records_available",
        _metric_value(artifacts, ["utr_records", "records"]),
        f">= {thresholds['min_utr_records']}",
        (_metric_value(artifacts, ["utr_records", "records"]) or 0) >= thresholds["min_utr_records"],
    )
    add_check(
        "skeleton_records_available",
        _metric_value(artifacts, ["skeleton_records", "records"]),
        f">= {thresholds['min_skeleton_records']}",
        (_metric_value(artifacts, ["skeleton_records", "records"]) or 0)
        >= thresholds["min_skeleton_records"],
    )
    add_check(
        "external_dify_workflows_available",
        _metric_value(payload, ["metrics", "external_dify_dataset", "workflow_count"]),
        f">= {thresholds['min_external_dify_workflows']}",
        (_metric_value(payload, ["metrics", "external_dify_dataset", "workflow_count"]) or 0)
        >= thresholds["min_external_dify_workflows"],
    )
    add_check(
        "external_supported_nodes_available",
        _metric_value(payload, ["metrics", "external_dify_dataset", "total_supported_nodes"]),
        f">= {thresholds['min_external_supported_nodes']}",
        (_metric_value(payload, ["metrics", "external_dify_dataset", "total_supported_nodes"]) or 0)
        >= thresholds["min_external_supported_nodes"],
    )

    internal_test_accuracy = _metric_value(
        payload, ["metrics", "node_mapping_internal", "test", "accuracy"]
    )
    add_check(
        "internal_node_mapping_test_accuracy",
        internal_test_accuracy,
        f">= {thresholds['min_internal_test_accuracy']}",
        internal_test_accuracy is not None
        and internal_test_accuracy >= thresholds["min_internal_test_accuracy"],
    )

    external_accuracy = _metric_value(payload, ["metrics", "node_mapping_external", "accuracy"])
    add_check(
        "external_node_mapping_accuracy",
        external_accuracy,
        f">= {thresholds['min_external_accuracy']}",
        external_accuracy is not None and external_accuracy >= thresholds["min_external_accuracy"],
    )

    internal_ece = _metric_value(
        payload, ["metrics", "node_mapping_internal", "test", "confidence_ece"]
    )
    if internal_ece is not None:
        add_check(
            "internal_node_mapping_confidence_ece",
            internal_ece,
            f"<= {thresholds['max_internal_confidence_ece']}",
            internal_ece <= thresholds["max_internal_confidence_ece"],
        )

    external_ece = _metric_value(payload, ["metrics", "node_mapping_external", "confidence_ece"])
    if external_ece is not None:
        add_check(
            "external_node_mapping_confidence_ece",
            external_ece,
            f"<= {thresholds['max_external_confidence_ece']}",
            external_ece <= thresholds["max_external_confidence_ece"],
        )

    commands = payload.get("commands", {})
    if "tests" in commands:
        tests = commands["tests"]
        pytest_summary = parse_pytest_summary(
            list(tests.get("stdout_tail", [])) + list(tests.get("stderr_tail", []))
        )
        tests["summary"] = pytest_summary
        add_check(
            "pytest_command_success",
            tests.get("returncode"),
            0,
            bool(tests.get("success")),
        )
        add_check(
            "pytest_no_failed_or_error_tests",
            pytest_summary,
            "failed == 0 and errors == 0",
            pytest_summary["failed"] == 0 and pytest_summary["errors"] == 0,
        )

    if "stage3_smoke" in commands:
        smoke = commands["stage3_smoke"]
        add_check(
            "stage3_smoke_command_success",
            smoke.get("returncode"),
            0,
            bool(smoke.get("success")),
        )
        add_check(
            "stage3_smoke_no_errors",
            smoke.get("error_count"),
            f"<= {thresholds['max_smoke_errors']}",
            smoke.get("error_count") is not None
            and smoke.get("error_count") <= thresholds["max_smoke_errors"],
        )
        add_check(
            "stage3_smoke_compiled_records",
            smoke.get("compiled_records"),
            "> 0",
            (smoke.get("compiled_records") or 0) > 0,
        )

    failed = [item for item in checks if not item["passed"] and item["severity"] == "error"]
    return {
        "thresholds": thresholds,
        "checks": checks,
        "passed": not failed,
        "failed_checks": [item["name"] for item in failed],
    }


def build_healthcheck(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": collect_artifacts(),
        "metrics": collect_metrics(),
        "commands": {},
    }
    if args.run_tests:
        payload["commands"]["tests"] = run_command([sys.executable, "-m", "pytest", "tests/", "-q"])
    if args.run_smoke:
        payload["commands"]["stage3_smoke"] = run_stage3_smoke()
    command_results = payload["commands"].values()
    payload["quality_gates"] = build_quality_gates(payload)
    commands_success = (
        all(item.get("success", False) for item in command_results) if payload["commands"] else True
    )
    payload["success"] = commands_success and payload["quality_gates"]["passed"]
    return payload


def main() -> None:
    args = parse_args()
    output_file = ROOT / args.output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = build_healthcheck(args)
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
