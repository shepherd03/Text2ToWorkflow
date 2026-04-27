import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.core.schema import (
    Action,
    NodeMappingEvalMetrics,
    NodeMappingEvalPrediction,
    NodeMappingEvalSample,
    NodeMappingEvalSummary,
)
from src.core.utils import append_to_jsonl
from src.dsl_generation.node_mapper import NodeMapper


EVAL_DIR = Path("generated_data/dsl_generation/node_mapping_eval")
TRAIN_PATH = EVAL_DIR / "train_samples.jsonl"
VALID_PATH = EVAL_DIR / "valid_samples.jsonl"
TEST_PATH = EVAL_DIR / "test_samples.jsonl"
HARD_PATH = EVAL_DIR / "hard_samples.jsonl"
RESULTS_JSONL = EVAL_DIR / "evaluation_predictions.jsonl"
SUMMARY_JSON = EVAL_DIR / "evaluation_summary.json"
BACKEND_SUMMARY_JSON = EVAL_DIR / "evaluation_summary_by_backend.json"
ERROR_ANALYSIS_JSON = EVAL_DIR / "error_analysis.json"

BACKEND_CONFIGS = {
    "tfidf": {
        "SEMANTIC_BACKEND": "tfidf",
        "SEMANTIC_EMBEDDING_PROVIDER": "local-hash",
        "SEMANTIC_EMBEDDING_API_KEY": "",
    },
    "embedding-local": {
        "SEMANTIC_BACKEND": "embedding",
        "SEMANTIC_EMBEDDING_PROVIDER": "local-hash",
        "SEMANTIC_EMBEDDING_API_KEY": "dummy",
    },
    "hybrid-local": {
        "SEMANTIC_BACKEND": "hybrid",
        "SEMANTIC_EMBEDDING_PROVIDER": "local-hash",
        "SEMANTIC_EMBEDDING_API_KEY": "dummy",
    },
}
FOCUS_LABELS = ["if-else", "iteration", "parameter-extractor", "variable-aggregator"]


def reset_outputs() -> None:
    if RESULTS_JSONL.exists():
        RESULTS_JSONL.unlink()


def load_samples(path: Path) -> list[NodeMappingEvalSample]:
    samples: list[NodeMappingEvalSample] = []
    if not path.exists():
        return samples
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        samples.append(NodeMappingEvalSample.model_validate(json.loads(line)))
    return samples


def collect_seen_signatures(train_samples: list[NodeMappingEvalSample]) -> set[str]:
    return {sample.text_signature for sample in train_samples if sample.text_signature}


def evaluate_samples(
    backend_name: str,
    mapper: NodeMapper,
    samples: list[NodeMappingEvalSample],
    seen_signatures: set[str],
) -> list[NodeMappingEvalPrediction]:
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
        prediction = NodeMappingEvalPrediction(
            sample_id=sample.sample_id,
            workflow_id=sample.workflow_id,
            backend=backend_name,
            source=sample.source,
            split=sample.split,
            expected_node_type=sample.expected_node_type,
            predicted_node_type=result.chosen_node_type,
            correct=result.chosen_node_type == sample.expected_node_type,
            expected_degraded=sample.expected_degraded,
            predicted_degraded=result.degraded,
            confidence=result.confidence,
            chosen_score=result.chosen_score,
            runner_up_score=result.runner_up_score,
            seen_in_train=sample.text_signature in seen_signatures,
            difficulty=sample.difficulty,
            tags=sample.tags,
        )
        predictions.append(prediction)
        append_to_jsonl(str(RESULTS_JSONL), prediction)

    return predictions


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


def build_confusion_matrix(
    predictions: list[NodeMappingEvalPrediction],
) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for prediction in predictions:
        matrix[prediction.expected_node_type.value][prediction.predicted_node_type.value] += 1
    return {
        expected: dict(sorted(predicted.items()))
        for expected, predicted in sorted(matrix.items())
    }


def compute_metrics(predictions: list[NodeMappingEvalPrediction]) -> NodeMappingEvalMetrics:
    if not predictions:
        return NodeMappingEvalMetrics()

    sample_count = len(predictions)
    accuracy = sum(1 for item in predictions if item.correct) / sample_count
    macro_f1 = compute_macro_f1(predictions)

    degrade_subset = [item for item in predictions if item.expected_degraded]
    degradation_accuracy = None
    if degrade_subset:
        degradation_accuracy = sum(
            1
            for item in degrade_subset
            if item.predicted_degraded and item.correct
        ) / len(degrade_subset)

    seen_subset = [item for item in predictions if item.seen_in_train]
    unseen_subset = [item for item in predictions if not item.seen_in_train]
    seen_accuracy = (
        sum(1 for item in seen_subset if item.correct) / len(seen_subset)
        if seen_subset
        else None
    )
    unseen_accuracy = (
        sum(1 for item in unseen_subset if item.correct) / len(unseen_subset)
        if unseen_subset
        else None
    )

    per_label_total = Counter(item.expected_node_type.value for item in predictions)
    per_label_correct = Counter(
        item.expected_node_type.value for item in predictions if item.correct
    )
    per_label_accuracy = {
        label: per_label_correct[label] / total
        for label, total in sorted(per_label_total.items())
    }

    return NodeMappingEvalMetrics(
        sample_count=sample_count,
        accuracy=accuracy,
        macro_f1=macro_f1,
        degradation_accuracy=degradation_accuracy,
        seen_accuracy=seen_accuracy,
        unseen_accuracy=unseen_accuracy,
        per_label_accuracy=per_label_accuracy,
    )


def build_summary(
    name: str,
    backend: str,
    predictions: list[NodeMappingEvalPrediction],
) -> NodeMappingEvalSummary:
    return NodeMappingEvalSummary(
        name=name,
        backend=backend,
        sample_count=len(predictions),
        metrics=compute_metrics(predictions),
        confusion_matrix=build_confusion_matrix(predictions),
    )


def run_single_backend(
    backend_name: str,
    train_samples: list[NodeMappingEvalSample],
    valid_samples: list[NodeMappingEvalSample],
    test_samples: list[NodeMappingEvalSample],
    hard_samples: list[NodeMappingEvalSample],
) -> dict:
    for key, value in BACKEND_CONFIGS[backend_name].items():
        os.environ[key] = value

    mapper = NodeMapper()
    seen_signatures = collect_seen_signatures(train_samples)

    valid_predictions = evaluate_samples(backend_name, mapper, valid_samples, seen_signatures)
    test_predictions = evaluate_samples(backend_name, mapper, test_samples, seen_signatures)
    hard_predictions = evaluate_samples(backend_name, mapper, hard_samples, seen_signatures)

    summary = {
        "backend": backend_name,
        "valid": build_summary("valid", backend_name, valid_predictions).model_dump(),
        "test": build_summary("test", backend_name, test_predictions).model_dump(),
        "hard": build_summary("hard", backend_name, hard_predictions).model_dump(),
        "predictions": {
            "valid": [prediction.model_dump() for prediction in valid_predictions],
            "test": [prediction.model_dump() for prediction in test_predictions],
            "hard": [prediction.model_dump() for prediction in hard_predictions],
        },
    }
    return summary


def build_comparison_table(summary_by_backend: dict[str, dict]) -> dict[str, dict[str, dict[str, float | None]]]:
    table: dict[str, dict[str, dict[str, float | None]]] = {}
    for split in ["valid", "test", "hard"]:
        table[split] = {}
        for backend_name, payload in summary_by_backend.items():
            metrics = payload[split]["metrics"]
            table[split][backend_name] = {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "degradation_accuracy": metrics["degradation_accuracy"],
                "seen_accuracy": metrics["seen_accuracy"],
                "unseen_accuracy": metrics["unseen_accuracy"],
            }
    return table


def build_error_analysis(summary_by_backend: dict[str, dict]) -> dict:
    analysis: dict[str, dict] = {}
    for backend_name, payload in summary_by_backend.items():
        backend_analysis: dict[str, dict] = {}
        for split in ["test", "hard"]:
            predictions = [
                NodeMappingEvalPrediction.model_validate(item)
                for item in payload["predictions"][split]
            ]
            split_analysis: dict[str, dict] = {}
            for label in FOCUS_LABELS:
                mistakes = [
                    prediction
                    for prediction in predictions
                    if prediction.expected_node_type.value == label and not prediction.correct
                ]
                confusion_counter = Counter(
                    prediction.predicted_node_type.value for prediction in mistakes
                )
                split_analysis[label] = {
                    "error_count": len(mistakes),
                    "top_confusions": dict(confusion_counter.most_common(3)),
                    "examples": [prediction.sample_id for prediction in mistakes[:5]],
                }
            backend_analysis[split] = split_analysis
        analysis[backend_name] = backend_analysis
    return analysis


def slim_summary(summary_by_backend: dict[str, dict]) -> dict:
    result: dict[str, dict] = {}
    for backend_name, payload in summary_by_backend.items():
        result[backend_name] = {
            "backend": backend_name,
            "valid": payload["valid"],
            "test": payload["test"],
            "hard": payload["hard"],
        }
    return result


def main() -> None:
    reset_outputs()
    train_samples = load_samples(TRAIN_PATH)
    valid_samples = load_samples(VALID_PATH)
    test_samples = load_samples(TEST_PATH)
    hard_samples = load_samples(HARD_PATH)
    if not train_samples or not valid_samples or not test_samples or not hard_samples:
        raise RuntimeError("Evaluation samples are missing, please run scripts/07_prepare_node_mapping_eval_data.py first.")

    summary_by_backend: dict[str, dict] = {}
    for backend_name in BACKEND_CONFIGS:
        summary_by_backend[backend_name] = run_single_backend(
            backend_name,
            train_samples,
            valid_samples,
            test_samples,
            hard_samples,
        )

    primary_backend = "tfidf"
    summary = slim_summary(summary_by_backend)[primary_backend]
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    BACKEND_SUMMARY_JSON.write_text(
        json.dumps(
            {
                "comparison": build_comparison_table(summary_by_backend),
                "backends": slim_summary(summary_by_backend),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    ERROR_ANALYSIS_JSON.write_text(
        json.dumps(build_error_analysis(summary_by_backend), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "comparison": build_comparison_table(summary_by_backend),
                "backends": slim_summary(summary_by_backend),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
