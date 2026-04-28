import json
import importlib.util
from pathlib import Path

from src.core.schema import DifyNodeType, NodeMappingEvalPrediction, NodeMappingEvalSample


def _load_module(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prepare_module = _load_module(
    "prepare_node_mapping_eval_data",
    "scripts/07_prepare_node_mapping_eval_data.py",
)
eval_module = _load_module(
    "evaluate_node_mapping_generalization",
    "scripts/08_evaluate_node_mapping_generalization.py",
)


def _sample(
    sample_id: str,
    expected: DifyNodeType,
    signature: str,
    degraded: bool = False,
) -> NodeMappingEvalSample:
    return NodeMappingEvalSample(
        sample_id=sample_id,
        workflow_id="wf_test",
        expected_node_type=expected,
        action_name=sample_id,
        description=f"description for {sample_id}",
        inputs=["url", "method"] if expected == DifyNodeType.http_request else ["draft"],
        outputs=["result"],
        available_resources=["tool_name", "provider_id"] if expected == DifyNodeType.tool else [],
        expected_degraded=degraded,
        text_signature=signature,
    )


def _prediction(
    expected: DifyNodeType,
    predicted: DifyNodeType,
    seen_in_train: bool,
    expected_degraded: bool = False,
    predicted_degraded: bool = False,
) -> NodeMappingEvalPrediction:
    return NodeMappingEvalPrediction(
        sample_id="sample",
        expected_node_type=expected,
        predicted_node_type=predicted,
        correct=expected == predicted,
        expected_degraded=expected_degraded,
        predicted_degraded=predicted_degraded,
        seen_in_train=seen_in_train,
    )


def test_split_samples_preserves_all_labels():
    samples = []
    for node_type in [
        DifyNodeType.llm,
        DifyNodeType.code,
        DifyNodeType.http_request,
    ]:
        for index in range(5):
            samples.append(
                _sample(
                    sample_id=f"{node_type.value}_{index}",
                    expected=node_type,
                    signature=f"sig_{node_type.value}_{index}",
                )
            )

    train, valid, test = prepare_module.split_samples(samples)

    assert train
    assert valid
    assert test
    assert {sample.expected_node_type for sample in train} == {
        DifyNodeType.llm,
        DifyNodeType.code,
        DifyNodeType.http_request,
    }


def test_build_hard_samples_contains_manual_cases():
    test_samples = [
        _sample("base_1", DifyNodeType.llm, "sig_base_1"),
        _sample("base_2", DifyNodeType.tool, "sig_base_2"),
    ]
    hard_samples = prepare_module.build_hard_samples(test_samples)

    ids = {sample.sample_id for sample in hard_samples}
    assert "hard::tool_missing" in ids
    assert "hard::http_missing" in ids
    assert any(sample.difficulty == "hard" for sample in hard_samples)


def test_compute_metrics_exposes_seen_unseen_and_degradation():
    predictions = [
        _prediction(DifyNodeType.llm, DifyNodeType.llm, True),
        _prediction(DifyNodeType.code, DifyNodeType.llm, True),
        _prediction(DifyNodeType.http_request, DifyNodeType.http_request, False),
        _prediction(
            DifyNodeType.code,
            DifyNodeType.code,
            False,
            expected_degraded=True,
            predicted_degraded=True,
        ),
    ]

    metrics = eval_module.compute_metrics(predictions)

    assert metrics.sample_count == 4
    assert metrics.accuracy == 0.75
    assert metrics.macro_f1 >= 0.0
    assert metrics.degradation_accuracy == 1.0
    assert metrics.degradation_detection_accuracy == 1.0
    assert metrics.degradation_type_accuracy == 1.0
    assert metrics.seen_accuracy == 0.5
    assert metrics.unseen_accuracy == 1.0
    assert "llm" in metrics.per_label_accuracy


def test_compute_metrics_separates_degradation_detection_from_type_accuracy():
    predictions = [
        _prediction(
            DifyNodeType.http_request,
            DifyNodeType.code,
            False,
            expected_degraded=True,
            predicted_degraded=True,
        ),
        _prediction(
            DifyNodeType.parameter_extractor,
            DifyNodeType.parameter_extractor,
            False,
            expected_degraded=True,
            predicted_degraded=False,
        ),
    ]

    metrics = eval_module.compute_metrics(predictions)

    assert metrics.degradation_accuracy == 0.0
    assert metrics.degradation_detection_accuracy == 0.5
    assert metrics.degradation_type_accuracy == 0.5


def test_build_summary_contains_confusion_matrix():
    predictions = [
        _prediction(DifyNodeType.llm, DifyNodeType.llm, True),
        _prediction(DifyNodeType.code, DifyNodeType.llm, False),
    ]
    summary = eval_module.build_summary("test", "tfidf", predictions)

    assert summary.name == "test"
    assert summary.backend == "tfidf"
    assert summary.confusion_matrix["llm"]["llm"] == 1
    assert summary.confusion_matrix["code"]["llm"] == 1


def test_manual_hard_case_helper_builds_hard_sample():
    sample = prepare_module.manual_hard_case(
        sample_id="hard::zh_case",
        expected_node_type=DifyNodeType.if_else,
        action_name="按风险级别分流",
        description="根据阈值决定是否人工复核",
        inputs=["conditions", "risk_score"],
        outputs=["route_result"],
        tags=["zh_long_tail"],
    )

    assert sample.split == "hard"
    assert sample.source == "hard"
    assert sample.difficulty == "hard"
    assert sample.expected_node_type == DifyNodeType.if_else
    assert "zh_long_tail" in sample.tags


def test_build_comparison_table_collects_backend_metrics():
    payload = {
        "tfidf": {
            "valid": {"metrics": {"accuracy": 0.4, "macro_f1": 0.3, "degradation_accuracy": None, "seen_accuracy": 0.5, "unseen_accuracy": 0.2}},
            "test": {"metrics": {"accuracy": 0.5, "macro_f1": 0.4, "degradation_accuracy": 0.0, "seen_accuracy": 0.3, "unseen_accuracy": 0.7}},
            "hard": {"metrics": {"accuracy": 0.6, "macro_f1": 0.5, "degradation_accuracy": 0.2, "seen_accuracy": None, "unseen_accuracy": 0.6}},
        },
        "hybrid-local": {
            "valid": {"metrics": {"accuracy": 0.45, "macro_f1": 0.35, "degradation_accuracy": None, "seen_accuracy": 0.55, "unseen_accuracy": 0.25}},
            "test": {"metrics": {"accuracy": 0.52, "macro_f1": 0.42, "degradation_accuracy": 0.1, "seen_accuracy": 0.33, "unseen_accuracy": 0.75}},
            "hard": {"metrics": {"accuracy": 0.58, "macro_f1": 0.48, "degradation_accuracy": 0.3, "seen_accuracy": None, "unseen_accuracy": 0.58}},
        },
    }

    table = eval_module.build_comparison_table(payload)

    assert table["test"]["tfidf"]["accuracy"] == 0.5
    assert table["hard"]["hybrid-local"]["macro_f1"] == 0.48


def test_build_error_analysis_extracts_focus_label_confusions():
    payload = {
        "tfidf": {
            "predictions": {
                "test": [
                    _prediction(DifyNodeType.if_else, DifyNodeType.code, False).model_dump(),
                    _prediction(DifyNodeType.if_else, DifyNodeType.code, False).model_dump(),
                    _prediction(DifyNodeType.iteration, DifyNodeType.llm, False).model_dump(),
                ],
                "hard": [
                    _prediction(DifyNodeType.variable_aggregator, DifyNodeType.code, False).model_dump(),
                ],
            }
        }
    }

    analysis = eval_module.build_error_analysis(payload)

    assert analysis["tfidf"]["test"]["if-else"]["error_count"] == 2
    assert analysis["tfidf"]["test"]["if-else"]["top_confusions"]["code"] == 2
    assert analysis["tfidf"]["hard"]["variable-aggregator"]["error_count"] == 1


def test_summary_json_can_roundtrip(tmp_path: Path):
    payload = {
        "backend": "tfidf",
        "valid": {"name": "valid", "backend": "tfidf", "sample_count": 1},
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["backend"] == "tfidf"
