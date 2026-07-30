from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from legal_qa_factory.blueprints.ranking import _question_similarity

PATTERN_FAMILIES = (
    "DIRECT_RULE",
    "CONDITION_EXCEPTION",
    "DEADLINE_CALCULATION",
    "PROCEDURE_DELEGATION",
    "SANCTION_REMEDY",
)

FAMILY_NEIGHBORS = {
    "DIRECT_RULE": {"CONDITION_EXCEPTION"},
    "CONDITION_EXCEPTION": {
        "DIRECT_RULE",
        "DEADLINE_CALCULATION",
        "PROCEDURE_DELEGATION",
    },
    "DEADLINE_CALCULATION": {"CONDITION_EXCEPTION"},
    "PROCEDURE_DELEGATION": {"CONDITION_EXCEPTION"},
    "SANCTION_REMEDY": {"PROCEDURE_DELEGATION"},
}


def inferred_family_hint(question: dict[str, Any]) -> str:
    if question["asks_sanction"]:
        return "SANCTION_REMEDY"
    if question["asks_deadline"]:
        return "DEADLINE_CALCULATION"
    if question["asks_procedure"]:
        return "PROCEDURE_DELEGATION"
    if question["asks_exception"] or question["asks_condition"]:
        return "CONDITION_EXCEPTION"
    return "DIRECT_RULE"


def family_grade(expected: str, candidate: str) -> int:
    if expected == candidate:
        return 3
    if candidate in FAMILY_NEIGHBORS[expected]:
        return 1
    return 0


def rank_families(
    query: dict[str, Any],
    training_rows: list[dict[str, Any]],
    *,
    k: int,
    neighbor_weight: float = 0.55,
    intent_weight: float = 0.45,
) -> list[dict[str, Any]]:
    neighbors = sorted(
        (
            (_question_similarity(query, row), row)
            for row in training_rows
        ),
        key=lambda item: (-item[0], item[1]["reference_qa_id"]),
    )[:k]
    votes: Counter[str] = Counter()
    normalizer = 0.0
    for similarity, row in neighbors:
        votes[row["pattern_family"]] += similarity
        normalizer += similarity
    hint = inferred_family_hint(query)
    ranked = []
    for family in PATTERN_FAMILIES:
        neighbor_score = votes[family] / normalizer if normalizer else 0.0
        hint_score = 1.0 if family == hint else 0.0
        if family in FAMILY_NEIGHBORS[hint]:
            hint_score = 0.25
        ranked.append(
            {
                "pattern_family": family,
                "score": (
                    neighbor_weight * neighbor_score
                    + intent_weight * hint_score
                ),
            }
        )
    return sorted(
        ranked,
        key=lambda row: (-row["score"], row["pattern_family"]),
    )


def _dcg(grades: list[int], k: int) -> float:
    return sum(
        (2**grade - 1) / math.log2(index + 2)
        for index, grade in enumerate(grades[:k])
    )


def _family_metrics(
    ranking: list[str], expected: str
) -> dict[str, float]:
    grades = [family_grade(expected, family) for family in ranking]
    ideal = sorted(grades, reverse=True)
    exact_rank = ranking.index(expected) + 1
    return {
        "family_recall@1": float(exact_rank <= 1),
        "family_recall@2": float(exact_rank <= 2),
        "family_recall@3": float(exact_rank <= 3),
        "family_mrr": 1 / exact_rank,
        "family_ndcg@3": _dcg(grades, 3) / _dcg(ideal, 3),
    }


def rank_components(
    training_rows: list[dict[str, Any]],
    neighbors: list[tuple[float, dict[str, Any]]],
    field: str,
) -> list[str]:
    universe = sorted(
        {
            value
            for row in training_rows
            for value in row[field]
        }
    )
    scores: Counter[str] = Counter()
    for similarity, row in neighbors:
        for value in set(row[field]):
            scores[value] += similarity
    return sorted(universe, key=lambda value: (-scores[value], value))


def _average_precision(
    ranking: list[str], expected: set[str]
) -> float:
    if not expected:
        return 1.0
    hits = 0
    precision_sum = 0.0
    for rank, value in enumerate(ranking, start=1):
        if value in expected:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(expected)


def evaluate_hierarchical_ranker(
    rows: list[dict[str, Any]], *, k: int
) -> dict[str, Any]:
    groups = sorted({row["parent_example_id"] for row in rows})
    metrics: dict[str, list[float]] = defaultdict(list)
    predictions = []
    for group_id in groups:
        training = [
            row for row in rows if row["parent_example_id"] != group_id
        ]
        held_out = [
            row for row in rows if row["parent_example_id"] == group_id
        ]
        for query in held_out:
            family_rows = rank_families(query, training, k=k)
            family_ranking = [
                row["pattern_family"] for row in family_rows
            ]
            family_result = _family_metrics(
                family_ranking, query["pattern_family"]
            )
            for name, value in family_result.items():
                metrics[name].append(value)

            neighbors = sorted(
                (
                    (_question_similarity(query, row), row)
                    for row in training
                ),
                key=lambda item: (-item[0], item[1]["reference_qa_id"]),
            )[:k]
            action_ranking = rank_components(
                training, neighbors, "retrieval_actions"
            )
            role_ranking = rank_components(
                training, neighbors, "answer_flow"
            )
            action_ap = _average_precision(
                action_ranking, set(query["retrieval_actions"])
            )
            role_ap = _average_precision(
                role_ranking, set(query["answer_flow"])
            )
            metrics["retrieval_action_map"].append(action_ap)
            metrics["answer_role_map"].append(role_ap)
            predictions.append(
                {
                    "reference_qa_id": query["reference_qa_id"],
                    "parent_example_id": group_id,
                    "expected_family": query["pattern_family"],
                    "family_ranking": family_ranking,
                    "retrieval_action_ranking": action_ranking,
                    "answer_role_ranking": role_ranking,
                }
            )
    return {
        "model_id": f"hierarchical_blueprint_ranker_v1_k{k}",
        "evaluation_strategy": "LEAVE_ONE_PARENT_GROUP_OUT",
        "physical_query_count": len(rows),
        "independent_group_count": len(groups),
        "metrics": {
            name: round(sum(values) / len(values), 4)
            for name, values in sorted(metrics.items())
        },
        "predictions": predictions,
    }

