import importlib.util
import json


def _load_module(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


healthcheck_module = _load_module(
    "project_healthcheck",
    "scripts/16_project_healthcheck.py",
)


def test_healthcheck_collects_artifacts_and_metrics(tmp_path):
    output_file = tmp_path / "health.json"
    args = healthcheck_module.parse_args(
        [
            "--output-file",
            str(output_file),
        ]
    )

    payload = healthcheck_module.build_healthcheck(args)
    output_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    assert payload["success"] is True
    assert payload["artifacts"]["utr_records"]["exists"] is True
    assert payload["artifacts"]["skeleton_records"]["exists"] is True
    assert "node_mapping_internal" in payload["metrics"]
    assert payload["quality_gates"]["passed"] is True
    assert payload["quality_gates"]["checks"]
    assert output_file.exists()


def test_healthcheck_parses_command_summaries():
    pytest_summary = healthcheck_module.parse_pytest_summary(
        ["147 passed, 2 skipped in 1.23s"]
    )
    smoke_summary = healthcheck_module.parse_stage3_smoke_summary(
        ["Completed. Success: 31, Errors: 0"]
    )

    assert pytest_summary["passed"] == 147
    assert pytest_summary["skipped"] == 2
    assert smoke_summary["compiled_records"] == 31
    assert smoke_summary["error_count"] == 0


def test_healthcheck_quality_gates_fail_on_missing_artifacts():
    payload = {
        "artifacts": {
            "utr_records": {"records": 0},
            "skeleton_records": {"records": 0},
        },
        "metrics": {
            "external_dify_dataset": {
                "workflow_count": 0,
                "total_supported_nodes": 0,
            },
            "node_mapping_internal": {
                "test": {"accuracy": 0.0, "confidence_ece": 0.4},
            },
            "node_mapping_external": {
                "accuracy": 0.0,
                "confidence_ece": 0.4,
            },
        },
        "commands": {},
    }

    gates = healthcheck_module.build_quality_gates(payload)

    assert gates["passed"] is False
    assert "utr_records_available" in gates["failed_checks"]
    assert "internal_node_mapping_confidence_ece" in gates["failed_checks"]
    assert "external_node_mapping_confidence_ece" in gates["failed_checks"]
