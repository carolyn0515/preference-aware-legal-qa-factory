from legal_qa_factory.blueprints.hierarchical import (
    family_grade,
    inferred_family_hint,
    rank_families,
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
