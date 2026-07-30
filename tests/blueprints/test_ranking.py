from legal_qa_factory.blueprints.ranking import (
    blueprint_relevance,
    lcs_similarity,
    ranking_metrics,
    weighted_jaccard,
)

CONFIG = {
    "relevance": {
        "answer_flow_weight": 0.55,
        "retrieval_action_weight": 0.45,
        "exact_match_grade": 3,
        "strong_match_threshold": 0.75,
        "partial_match_threshold": 0.4,
    }
}


def test_component_similarities_are_bounded() -> None:
    assert weighted_jaccard(["A", "B"], ["A", "C"]) == 1 / 3
    assert lcs_similarity(["A", "B", "C"], ["A", "C"]) == 2 / 3


def test_exact_blueprint_receives_maximum_grade() -> None:
    blueprint = {
        "pattern_id": "P1",
        "answer_flow": ["RULE", "CHECK"],
        "retrieval_actions": ["ANCHOR"],
    }
    score, grade = blueprint_relevance(blueprint, blueprint, CONFIG)
    assert score == 1.0
    assert grade == 3


def test_ranking_metrics_reward_near_top_exact_match() -> None:
    metrics = ranking_metrics(
        ["P2", "P1", "P3"],
        {"P1": 3, "P2": 2, "P3": 0},
        "P1",
        [3],
        [1, 3],
    )
    assert metrics["reciprocal_rank"] == 0.5
    assert metrics["recall@1"] == 0.0
    assert metrics["recall@3"] == 1.0
    assert 0 < metrics["ndcg@3"] < 1
