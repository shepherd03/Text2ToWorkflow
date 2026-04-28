import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.core.schema import (
    Action,
    NodeMappingEvalPrediction,
    NodeMappingEvalSample,
)
from src.core.utils import append_to_jsonl
from src.dsl_generation.node_mapper import NodeMapper


EVAL_DIR = Path("generated_data/dsl_generation/external_node_mapping_eval")
SAMPLES_PATH = EVAL_DIR / "external_samples.jsonl"
RESULTS_PATH = EVAL_DIR / "external_predictions.jsonl"
SUMMARY_PATH = EVAL_DIR / "external_evaluation_summary.json"

BACKEND_CONFIG = {
    "SEMANTIC_BACKEND": "tfidf",
    "SEMANTIC_EMBEDDING_PROVIDER": "local-hash",
    "SEMANTIC_EMBEDDING_API_KEY": "",
}


def load_samples() -> list[NodeMappingEvalSample]:
    samples: list[NodeMappingEvalSample] = []
    for line in SAMPLES_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        samples.append(NodeMappingEvalSample.model_validate(json.loads(line)))
    return samples


def compute_macro_f1(predictions: list[NodeMappingEvalPrediction]) -> float:
    labels = sorted(
        {
            prediction.expected_node_type.value
            for prediction in predictions
        }
        | {
            prediction.predicted_node_type.value
            for prediction in predictions
        }
    )
    if not labels:
        return 0.0

    f1_scores: list[float] = []
    for label in labels:
        tp = sum(
            1
            for prediction in predictions
            if prediction.expected_node_type.value == label
            and prediction.predicted_node_type.value == label
        )
        fp = sum(
            1
            for prediction in predictions
            if prediction.expected_node_type.value != label
            and prediction.predicted_node_type.value == label
        )
        fn = sum(
            1
            for prediction in predictions
            if prediction.expected_node_type.value == label
            and prediction.predicted_node_type.value != label
        )
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores)


def build_confusion_matrix(predictions: list[NodeMappingEvalPrediction]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for prediction in predictions:
        expected = prediction.expected_node_type.value
        predicted = prediction.predicted_node_type.value
        matrix.setdefault(expected, {})
        matrix[expected][predicted] = matrix[expected].get(predicted, 0) + 1
    return {
        expected: dict(sorted(predicted.items()))
        for expected, predicted in sorted(matrix.items())
    }


def compute_confidence_calibration(
    predictions: list[NodeMappingEvalPrediction],
) -> dict[str, float | dict[str, dict[str, float | int]]]:
    if not predictions:
        return {
            "confidence_ece": 0.0,
            "confidence_brier": 0.0,
            "confidence_bucket_accuracy": {},
        }
    buckets = [
        ("0.00-0.50", 0.0, 0.50),
        ("0.50-0.70", 0.50, 0.70),
        ("0.70-0.85", 0.70, 0.85),
        ("0.85-1.00", 0.85, 1.000001),
    ]
    bucket_accuracy: dict[str, dict[str, float | int]] = {}
    ece = 0.0
    brier = 0.0
    total = len(predictions)
    for prediction in predictions:
        target = 1.0 if prediction.correct else 0.0
        score = min(max(prediction.confidence_score, 0.0), 1.0)
        brier += (score - target) ** 2
    for name, lower, upper in buckets:
        bucket_items = [
            prediction
            for prediction in predictions
            if lower <= prediction.confidence_score < upper
        ]
        if not bucket_items:
            continue
        accuracy = sum(1 for item in bucket_items if item.correct) / len(bucket_items)
        avg_confidence = sum(item.confidence_score for item in bucket_items) / len(bucket_items)
        ece += (len(bucket_items) / total) * abs(accuracy - avg_confidence)
        bucket_accuracy[name] = {
            "sample_count": len(bucket_items),
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
        }
    return {
        "confidence_ece": ece,
        "confidence_brier": brier / total,
        "confidence_bucket_accuracy": bucket_accuracy,
    }


def evaluate(samples: list[NodeMappingEvalSample]) -> tuple[list[NodeMappingEvalPrediction], dict]:
    for key, value in BACKEND_CONFIG.items():
        os.environ[key] = value

    if RESULTS_PATH.exists():
        RESULTS_PATH.unlink()

    mapper = NodeMapper()
    predictions: list[NodeMappingEvalPrediction] = []

    for sample in samples:
        action = Action(
            action_id=sample.sample_id,
            action_name=sample.action_name,
            description=sample.description,
            inputs=sample.inputs,
            outputs=sample.outputs,
        )
        result = mapper.map_action(
            action,
            parent_block_type=sample.parent_block_type,
            available_resources=sample.available_resources,
        )
        confidence_score = mapper.confidence_probability(result)
        prediction = NodeMappingEvalPrediction(
            sample_id=sample.sample_id,
            workflow_id=sample.workflow_id,
            backend="tfidf",
            source=sample.source,
            split=sample.split,
            expected_node_type=sample.expected_node_type,
            predicted_node_type=result.chosen_node_type,
            correct=result.chosen_node_type == sample.expected_node_type,
            expected_degraded=sample.expected_degraded,
            predicted_degraded=result.degraded,
            confidence=result.confidence,
            confidence_score=confidence_score,
            confidence_margin=result.chosen_score - result.runner_up_score,
            chosen_score=result.chosen_score,
            runner_up_score=result.runner_up_score,
            seen_in_train=False,
            difficulty=sample.difficulty,
            tags=sample.tags,
        )
        predictions.append(prediction)
        append_to_jsonl(str(RESULTS_PATH), prediction)

    per_label_total: dict[str, int] = {}
    per_label_correct: dict[str, int] = {}
    degraded_total = 0
    degraded_correct = 0
    for prediction in predictions:
        label = prediction.expected_node_type.value
        per_label_total[label] = per_label_total.get(label, 0) + 1
        if prediction.correct:
            per_label_correct[label] = per_label_correct.get(label, 0) + 1
        if prediction.expected_degraded:
            degraded_total += 1
            if prediction.correct and prediction.predicted_degraded:
                degraded_correct += 1

    accuracy = sum(1 for item in predictions if item.correct) / len(predictions)
    calibration_metrics = compute_confidence_calibration(predictions)
    summary = {
        "backend": "tfidf",
        "sample_count": len(predictions),
        "accuracy": accuracy,
        "macro_f1": compute_macro_f1(predictions),
        "degradation_accuracy": degraded_correct / degraded_total if degraded_total else None,
        **calibration_metrics,
        "per_label_accuracy": {
            label: per_label_correct.get(label, 0) / total
            for label, total in sorted(per_label_total.items())
        },
        "confusion_matrix": build_confusion_matrix(predictions),
    }
    return predictions, summary


def main() -> None:
    samples = load_samples()
    _, summary = evaluate(samples)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
