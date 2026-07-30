from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from legal_qa_factory.blueprints.registry import _cosine


def weighted_jaccard(
    left: list[str],
    right: list[str],
    item_weights: dict[str, float] | None = None,
) -> float:
    weights = item_weights or {}
    union = set(left) | set(right)
    if not union:
        return 1.0
    intersection = set(left) & set(right)
    numerator = sum(weights.get(item, 1.0) for item in intersection)
    denominator = sum(weights.get(item, 1.0) for item in union)
    return numerator / denominator if denominator else 0.0


def lcs_similarity(left: list[str], right: list[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for index, right_item in enumerate(right, start=1):
            if left_item == right_item:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1] / max(len(left), len(right))


def blueprint_relevance(
    gold: dict[str, Any],
    candidate: dict[str, Any],
    config: dict[str, Any],
) -> tuple[float, int]:
    settings = config["relevance"]
    flow_score = lcs_similarity(
        gold["answer_flow"], candidate["answer_flow"]
    )
    action_score = weighted_jaccard(
        gold["retrieval_actions"], candidate["retrieval_actions"]
    )
    score = (
        settings["answer_flow_weight"] * flow_score
        + settings["retrieval_action_weight"] * action_score
    )
    if gold["pattern_id"] == candidate["pattern_id"]:
        grade = settings["exact_match_grade"]
        score = 1.0
    elif score >= settings["strong_match_threshold"]:
        grade = 2
    elif score >= settings["partial_match_threshold"]:
        grade = 1
    else:
        grade = 0
    return round(score, 6), grade


def candidate_registry(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    patterns: dict[str, dict[str, Any]] = {}
    for row in rows:
        patterns.setdefault(
            row["pattern_id"],
            {
                "pattern_id": row["pattern_id"],
                "answer_flow": row["answer_flow"],
                "retrieval_actions": row["retrieval_actions"],
            },
        )
    return [patterns[key] for key in sorted(patterns)]


def compile_ranking_rows(
    policy_rows: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = candidate_registry(policy_rows)
    ranking_rows = []
    for query in policy_rows:
        for candidate in candidates:
            score, grade = blueprint_relevance(query, candidate, config)
            ranking_rows.append(
                {
                    "reference_qa_id": query["reference_qa_id"],
                    "parent_example_id": query["parent_example_id"],
                    "candidate_pattern_id": candidate["pattern_id"],
                    "candidate_answer_flow": candidate["answer_flow"],
                    "candidate_retrieval_actions": candidate[
                        "retrieval_actions"
                    ],
                    "relevance_score": score,
                    "relevance_grade": grade,
                    "is_exact_pattern": (
                        query["pattern_id"] == candidate["pattern_id"]
                    ),
                }
            )
    return ranking_rows, candidates


def _question_similarity(
    query: dict[str, Any], neighbor: dict[str, Any]
) -> float:
    lexical = _cosine(query["question_terms"], neighbor["question_terms"])
    query_intents = set(query["question_intents"])
    neighbor_intents = set(neighbor["question_intents"])
    intent = len(query_intents & neighbor_intents) / max(
        len(query_intents | neighbor_intents), 1
    )
    return 0.8 * lexical + 0.2 * intent


def rank_candidates(
    query: dict[str, Any],
    training_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    k: int,
) -> list[dict[str, Any]]:
    neighbors = sorted(
        (
            (_question_similarity(query, row), row)
            for row in training_rows
        ),
        key=lambda item: (-item[0], item[1]["reference_qa_id"]),
    )[:k]
    scores: dict[str, float] = defaultdict(float)
    normalizer = sum(similarity for similarity, _ in neighbors)
    for candidate in candidates:
        for similarity, neighbor in neighbors:
            relevance, _ = blueprint_relevance(neighbor, candidate, config)
            scores[candidate["pattern_id"]] += similarity * relevance
        if normalizer:
            scores[candidate["pattern_id"]] /= normalizer
    return sorted(
        (
            {
                **candidate,
                "predicted_relevance": round(
                    scores[candidate["pattern_id"]], 6
                ),
            }
            for candidate in candidates
        ),
        key=lambda row: (-row["predicted_relevance"], row["pattern_id"]),
    )


def dcg(grades: list[int], k: int) -> float:
    return sum(
        (2**grade - 1) / math.log2(rank + 2)
        for rank, grade in enumerate(grades[:k])
    )


def ranking_metrics(
    ranked_pattern_ids: list[str],
    grades_by_pattern: dict[str, int],
    exact_pattern_id: str,
    ndcg_values: list[int],
    recall_values: list[int],
) -> dict[str, float]:
    ranked_grades = [
        grades_by_pattern[pattern_id] for pattern_id in ranked_pattern_ids
    ]
    ideal_grades = sorted(grades_by_pattern.values(), reverse=True)
    exact_rank = ranked_pattern_ids.index(exact_pattern_id) + 1
    result: dict[str, float] = {
        "reciprocal_rank": 1 / exact_rank,
        "exact_rank": float(exact_rank),
    }
    for k in ndcg_values:
        ideal = dcg(ideal_grades, k)
        result[f"ndcg@{k}"] = dcg(ranked_grades, k) / ideal if ideal else 0.0
    for k in recall_values:
        result[f"recall@{k}"] = float(exact_rank <= k)
    return result


def evaluate_grouped_ranker(
    policy_rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    k: int,
) -> dict[str, Any]:
    candidates = candidate_registry(policy_rows)
    group_ids = sorted({row["parent_example_id"] for row in policy_rows})
    metric_rows = []
    predictions = []
    for group_id in group_ids:
        training = [
            row
            for row in policy_rows
            if row["parent_example_id"] != group_id
        ]
        held_out = [
            row
            for row in policy_rows
            if row["parent_example_id"] == group_id
        ]
        for query in held_out:
            ranked = rank_candidates(
                query, training, candidates, config, k=k
            )
            grades = {
                candidate["pattern_id"]: blueprint_relevance(
                    query, candidate, config
                )[1]
                for candidate in candidates
            }
            metrics = ranking_metrics(
                [row["pattern_id"] for row in ranked],
                grades,
                query["pattern_id"],
                config["evaluation"]["ndcg_k"],
                config["evaluation"]["recall_k"],
            )
            metric_rows.append(metrics)
            predictions.append(
                {
                    "reference_qa_id": query["reference_qa_id"],
                    "parent_example_id": group_id,
                    "expected_pattern_id": query["pattern_id"],
                    "top_pattern_id": ranked[0]["pattern_id"],
                    "exact_rank": int(metrics["exact_rank"]),
                    "top_candidates": [
                        {
                            "pattern_id": row["pattern_id"],
                            "predicted_relevance": row[
                                "predicted_relevance"
                            ],
                            "gold_grade": grades[row["pattern_id"]],
                        }
                        for row in ranked[:5]
                    ],
                }
            )
    metric_names = sorted(metric_rows[0])
    aggregate = {
        name: round(
            sum(row[name] for row in metric_rows) / len(metric_rows), 4
        )
        for name in metric_names
    }
    return {
        "model_id": f"{config['ranker_id']}_k{k}",
        "evaluation_strategy": config["evaluation"]["split_strategy"],
        "physical_query_count": len(policy_rows),
        "independent_group_count": len(group_ids),
        "candidate_count": len(candidates),
        "metrics": aggregate,
        "predictions": predictions,
    }

