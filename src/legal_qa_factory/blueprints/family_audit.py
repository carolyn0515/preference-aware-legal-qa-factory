from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from legal_qa_factory.blueprints.hierarchical import (
    inferred_family_hint,
    rank_families,
)


def _matrix(
    pairs: list[tuple[str, str]]
) -> dict[str, dict[str, int]]:
    labels = sorted(
        {expected for expected, _ in pairs}
        | {predicted for _, predicted in pairs}
    )
    counts = Counter(pairs)
    return {
        expected: {
            predicted: counts[(expected, predicted)]
            for predicted in labels
        }
        for expected in labels
    }


def audit_family_predictions(
    rows: list[dict[str, Any]], *, k: int = 3
) -> dict[str, Any]:
    group_ids = sorted({row["parent_example_id"] for row in rows})
    row_results = []
    group_predictions: dict[str, list[str]] = defaultdict(list)
    group_expected: dict[str, str] = {}
    for group_id in group_ids:
        training = [
            row for row in rows if row["parent_example_id"] != group_id
        ]
        held_out = [
            row for row in rows if row["parent_example_id"] == group_id
        ]
        expected_values = {row["pattern_family"] for row in held_out}
        if len(expected_values) != 1:
            raise ValueError(f"inconsistent family labels in {group_id}")
        expected = next(iter(expected_values))
        group_expected[group_id] = expected
        for row in held_out:
            hint = inferred_family_hint(row)
            predicted = rank_families(row, training, k=k)[0][
                "pattern_family"
            ]
            if predicted == expected:
                error_type = "CORRECT"
            elif hint != expected:
                error_type = "QUESTION_TARGET_MISMATCH"
            else:
                error_type = "NEIGHBOR_OVERRIDE"
            group_predictions[group_id].append(predicted)
            row_results.append(
                {
                    "reference_qa_id": row["reference_qa_id"],
                    "parent_example_id": group_id,
                    "reference_topic": row["reference_topic"],
                    "reference_intent": row["reference_intent"],
                    "generation_variant_id": row[
                        "generation_variant_id"
                    ],
                    "expected_family": expected,
                    "inferred_hint": hint,
                    "predicted_family": predicted,
                    "correct": predicted == expected,
                    "error_type": error_type,
                    "question": row["question"],
                }
            )

    group_results = []
    for group_id in group_ids:
        votes = Counter(group_predictions[group_id])
        predicted = max(votes, key=lambda family: (votes[family], family))
        expected = group_expected[group_id]
        example = next(
            row
            for row in row_results
            if row["parent_example_id"] == group_id
        )
        group_results.append(
            {
                "parent_example_id": group_id,
                "reference_topic": example["reference_topic"],
                "reference_intent": example["reference_intent"],
                "expected_family": expected,
                "predicted_family": predicted,
                "correct": predicted == expected,
                "variant_vote_counts": dict(sorted(votes.items())),
            }
        )

    row_pairs = [
        (row["expected_family"], row["predicted_family"])
        for row in row_results
    ]
    group_pairs = [
        (row["expected_family"], row["predicted_family"])
        for row in group_results
    ]
    variant_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in row_results:
        variant_counts[row["generation_variant_id"]]["total"] += 1
        variant_counts[row["generation_variant_id"]][
            "correct" if row["correct"] else "incorrect"
        ] += 1
    variant_summary = {}
    for variant, counts in sorted(variant_counts.items()):
        variant_summary[variant] = {
            **dict(counts),
            "accuracy": round(counts["correct"] / counts["total"], 4),
        }
    errors = Counter(
        row["error_type"] for row in row_results if not row["correct"]
    )
    return {
        "schema_version": "1.0",
        "audit_method": "LEAVE_ONE_PARENT_GROUP_OUT_FAMILY_AUDIT",
        "k": k,
        "physical_row_count": len(rows),
        "independent_group_count": len(group_ids),
        "row_accuracy": round(
            sum(row["correct"] for row in row_results) / len(row_results),
            4,
        ),
        "group_accuracy": round(
            sum(row["correct"] for row in group_results)
            / len(group_results),
            4,
        ),
        "error_type_counts": dict(sorted(errors.items())),
        "variant_summary": variant_summary,
        "row_confusion_matrix": _matrix(row_pairs),
        "group_confusion_matrix": _matrix(group_pairs),
        "group_results": group_results,
        "misclassified_rows": [
            row for row in row_results if not row["correct"]
        ],
    }

