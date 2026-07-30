from __future__ import annotations

from collections import Counter
from typing import Any

from legal_qa_factory.blueprints.registry import recommend_knn

MINIMUM_EXAMPLES = 20
MINIMUM_CLASS_SUPPORT = 5


def classification_metrics(
    truth: list[str], predictions: list[str]
) -> dict[str, Any]:
    if len(truth) != len(predictions):
        raise ValueError("truth and predictions must have equal length")
    labels = sorted(set(truth) | set(predictions))
    report = {}
    for label in labels:
        true_positive = sum(
            expected == label and predicted == label
            for expected, predicted in zip(truth, predictions, strict=True)
        )
        false_positive = sum(
            expected != label and predicted == label
            for expected, predicted in zip(truth, predictions, strict=True)
        )
        support = sum(expected == label for expected in truth)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        report[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1-score": round(f1, 4),
            "support": support,
        }
    supported = [report[label] for label in labels if report[label]["support"]]
    total = len(truth)
    accuracy = (
        sum(
            expected == predicted
            for expected, predicted in zip(truth, predictions, strict=True)
        )
        / total
        if total
        else 0.0
    )
    macro_f1 = (
        sum(item["f1-score"] for item in supported) / len(supported)
        if supported
        else 0.0
    )
    weighted_f1 = (
        sum(item["f1-score"] * item["support"] for item in supported) / total
        if total
        else 0.0
    )
    return {
        "per_class": report,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "example_count": total,
    }


def reportability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    supports = Counter(row["pattern_id"] for row in rows)
    blockers = []
    if len(rows) < MINIMUM_EXAMPLES:
        blockers.append("MINIMUM_EXAMPLES_NOT_MET")
    if len(supports) < 2:
        blockers.append("MINIMUM_PATTERN_COUNT_NOT_MET")
    singleton = sorted(
        pattern_id
        for pattern_id, support in supports.items()
        if support < MINIMUM_CLASS_SUPPORT
    )
    if singleton:
        blockers.append("MINIMUM_CLASS_SUPPORT_NOT_MET")
    if not all(row["production_training_eligible"] for row in rows):
        blockers.append("NON_PRODUCTION_LABELS_PRESENT")
    return {
        "status": "NOT_REPORTABLE" if blockers else "REPORTABLE",
        "minimum_examples": MINIMUM_EXAMPLES,
        "minimum_class_support": MINIMUM_CLASS_SUPPORT,
        "class_support": dict(sorted(supports.items())),
        "under_supported_patterns": singleton,
        "blockers": blockers,
    }


def leave_one_out_knn(
    rows: list[dict[str, Any]], *, k: int
) -> dict[str, Any]:
    truth, predictions = [], []
    prediction_rows = []
    for index, held_out in enumerate(rows):
        training = rows[:index] + rows[index + 1 :]
        if not training:
            raise ValueError("leave-one-out evaluation requires at least 2 rows")
        result = recommend_knn(
            held_out["question"], training, k=min(k, len(training))
        )
        truth.append(held_out["pattern_id"])
        predictions.append(result["pattern_id"])
        prediction_rows.append(
            {
                "reference_qa_id": held_out["reference_qa_id"],
                "expected_pattern_id": held_out["pattern_id"],
                "predicted_pattern_id": result["pattern_id"],
                "correct": held_out["pattern_id"] == result["pattern_id"],
                "maximum_similarity": result["maximum_similarity"],
                "vote_share": result["vote_share"],
            }
        )
    return {
        "model_id": f"weighted_knn_k{k}_v1",
        "evaluation_strategy": "LEAVE_ONE_OUT",
        "metrics": classification_metrics(truth, predictions),
        "predictions": prediction_rows,
    }


def benchmark_policy_models(
    rows: list[dict[str, Any]], k_values: tuple[int, ...] = (1, 3, 5)
) -> dict[str, Any]:
    gate = reportability(rows)
    effective_k = sorted({min(k, len(rows) - 1) for k in k_values if k > 0})
    models = [leave_one_out_knn(rows, k=k) for k in effective_k]
    leader = max(
        models,
        key=lambda model: (
            model["metrics"]["macro_f1"],
            model["metrics"]["accuracy"],
            model["model_id"],
        ),
    )
    return {
        "schema_version": "1.0",
        "status": gate["status"],
        "selection_status": (
            "SELECTED" if gate["status"] == "REPORTABLE" else "NOT_SELECTED"
        ),
        "best_model_id": (
            leader["model_id"] if gate["status"] == "REPORTABLE" else None
        ),
        "diagnostic_leader_id": leader["model_id"],
        "reportability": gate,
        "models": models,
        "warning": (
            None
            if gate["status"] == "REPORTABLE"
            else (
                "Metrics are diagnostic only. Do not describe the diagnostic "
                "leader as the best-performing model."
            )
        ),
    }
