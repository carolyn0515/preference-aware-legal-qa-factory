from legal_qa_factory.blueprints.hierarchical import (
    family_grade,
    inferred_family_hint,
    rank_families,
    recommend_hierarchical,
)


def question(**overrides):
    row = {
        "reference_qa_id": "Q1",
        "question_terms": ["과징금"],
        "question_intents": ["SANCTION"],
        "asks_condition": False,
        "asks_exception": False,
        "asks_procedure": False,
        "asks_deadline": False,
        "asks_sanction": True,
    }
    row.update(overrides)
    return row


def test_sanction_intent_maps_to_sanction_family() -> None:
    assert inferred_family_hint(question()) == "SANCTION_REMEDY"
    assert family_grade("SANCTION_REMEDY", "SANCTION_REMEDY") == 3


def test_family_ranker_can_rank_unseen_family_from_intent() -> None:
    training = [
        {
            **question(
                reference_qa_id="TRAIN",
                question_terms=["대금"],
                question_intents=["GENERAL_LEGAL_QA"],
                asks_sanction=False,
            ),
            "pattern_family": "DIRECT_RULE",
        }
    ]
    ranked = rank_families(question(), training, k=1)

    assert ranked[0]["pattern_family"] == "SANCTION_REMEDY"


def test_recommender_returns_two_executable_blueprint_branches() -> None:
    rows = []
    for index, family in enumerate(
        ("DIRECT_RULE", "CONDITION_EXCEPTION"), start=1
    ):
        rows.append(
            {
                **question(
                    reference_qa_id=f"Q{index}",
                    question_terms=["대금", str(index)],
                    question_intents=["GENERAL_LEGAL_QA"],
                    asks_sanction=False,
                ),
                "parent_example_id": f"G{index}",
                "pattern_family": family,
                "retrieval_actions": ["SEARCH_ANCHOR"],
                "answer_flow": ["CONCLUSION", "PRACTICAL_GUIDANCE"],
            }
        )

    result = recommend_hierarchical(
        "하도급대금은 언제 지급해야 하나요?",
        rows,
        k=2,
        top_families=2,
    )

    assert result["routing_policy"] == "EXECUTE_TOP_2_THEN_EVIDENCE_RERANK"
    assert len(result["blueprints"]) == 2
    assert all(
        item["selected_retrieval_actions"] for item in result["blueprints"]
    )
