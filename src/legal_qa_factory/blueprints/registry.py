from __future__ import annotations

import math
from collections import Counter
from typing import Any

from legal_qa_factory.blueprints.compiler import question_features


def _cosine(left: list[str], right: list[str]) -> float:
    left_counts, right_counts = Counter(left), Counter(right)
    common = set(left_counts) & set(right_counts)
    numerator = sum(left_counts[key] * right_counts[key] for key in common)
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    right_norm = math.sqrt(
        sum(value * value for value in right_counts.values())
    )
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def recommend_knn(
    question: str,
    training_rows: list[dict[str, Any]],
    *,
    k: int = 3,
) -> dict[str, Any]:
    if not training_rows:
        raise ValueError("policy training dataset is empty")
    query = question_features(question)
    boolean_fields = (
        "has_explicit_citation",
        "asks_condition",
        "asks_exception",
        "asks_procedure",
        "asks_deadline",
        "asks_sanction",
    )
    neighbors = []
    for row in training_rows:
        lexical = _cosine(query["question_terms"], row["question_terms"])
        intent_overlap = len(
            set(query["question_intents"]) & set(row["question_intents"])
        ) / max(len(set(query["question_intents"])), 1)
        boolean_agreement = sum(
            query[field] == row[field] for field in boolean_fields
        ) / len(boolean_fields)
        score = 0.65 * lexical + 0.20 * intent_overlap + 0.15 * boolean_agreement
        neighbors.append({**row, "similarity": score})
    neighbors.sort(
        key=lambda row: (-row["similarity"], row["reference_qa_id"])
    )
    selected = neighbors[: min(k, len(neighbors))]
    votes: Counter[str] = Counter()
    for row in selected:
        votes[row["pattern_id"]] += row["similarity"] * row["sample_weight"]
    pattern_id, vote = max(votes.items(), key=lambda item: (item[1], item[0]))
    winner = next(row for row in selected if row["pattern_id"] == pattern_id)
    total_vote = sum(votes.values())
    vote_share = vote / total_vote if total_vote else 0.0
    maximum_similarity = selected[0]["similarity"]
    status = (
        "EXPERIMENTAL_RECOMMENDATION"
        if not winner["production_training_eligible"]
        else "RECOMMENDATION"
    )
    if maximum_similarity < 0.25:
        status = "ABSTAIN_OUT_OF_DISTRIBUTION"
    return {
        "method": "WEIGHTED_KNN_BASELINE_V1",
        "status": status,
        "question_features": query,
        "pattern_id": pattern_id,
        "answer_flow": winner["answer_flow"],
        "retrieval_actions": winner["retrieval_actions"],
        "vote_share": round(vote_share, 4),
        "maximum_similarity": round(maximum_similarity, 4),
        "confidence_calibrated": False,
        "neighbors": [
            {
                "reference_qa_id": row["reference_qa_id"],
                "pattern_id": row["pattern_id"],
                "similarity": round(row["similarity"], 4),
            }
            for row in selected
        ],
    }
