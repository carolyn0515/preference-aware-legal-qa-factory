from legal_qa_factory.retrieval.blueprint_executor import (
    _allowed_relations,
    _score_evidence,
)


def test_blueprint_actions_limit_tree_relations() -> None:
    assert _allowed_relations(
        ["SEARCH_ANCHOR", "EXPAND_CHILDREN"]
    ) == {"CHILD_ENUMERATION"}


def test_evidence_score_exposes_independent_components() -> None:
    score = _score_evidence(
        question="대금 지급",
        text="하도급대금을 지급한다",
        question_relevance=0.8,
        hierarchy_proximity=1.0,
        weights={
            "question_relevance": 0.5,
            "query_coverage": 0.3,
            "hierarchy_proximity": 0.2,
        },
    )

    assert set(score) == {
        "question_relevance",
        "query_coverage",
        "hierarchy_proximity",
        "final_score",
    }
    assert 0 <= score["final_score"] <= 1
