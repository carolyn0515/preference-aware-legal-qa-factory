from legal_qa_factory.blueprints.compiler import (
    abstract_retrieval_actions,
    compile_training_rows,
    question_features,
    routing_family,
)
from legal_qa_factory.blueprints.registry import recommend_knn
from legal_qa_factory.blueprints.validator import training_gate


def test_question_features_do_not_require_answer() -> None:
    result = question_features("직접지급 사유와 절차는 무엇인가요?")
    assert result["asks_condition"] is True
    assert result["asks_procedure"] is True
    assert "DEFINITION" in result["question_intents"]


def test_traversal_is_abstracted_into_reusable_actions() -> None:
    result = abstract_retrieval_actions(
        [
            "ANCHOR:ACT:제14조",
            "CHILD_ENUMERATION:ACT:제14조:1호",
            "IMPLEMENTING_DECREE:DECREE:제9조:①",
        ]
    )
    assert result == [
        "SEARCH_ANCHOR",
        "EXPAND_CHILDREN",
        "FOLLOW_DECREE_DELEGATION",
    ]


def test_routing_family_uses_question_observable_intent() -> None:
    assert (
        routing_family("prohibition", "CONDITION_EXCEPTION")
        == "DIRECT_RULE"
    )
    assert (
        routing_family("deadline", "CONDITION_EXCEPTION")
        == "DEADLINE_CALCULATION"
    )


def test_synthetic_small_dataset_is_blocked_but_knn_can_demo() -> None:
    qa = {
        "reference_qa_id": "Q1",
        "customer_id": "C",
        "reference_version": "v1",
        "source_kind": "SYNTHETIC",
        "question": "직접지급 사유는 무엇인가요?",
    }
    rows, _ = compile_training_rows(
        [qa],
        [{"reference_qa_id": "Q1", "answer_flow": ["CONDITION"]}],
        [
            {
                "reference_qa_id": "Q1",
                "traversal_flow": [
                    "ANCHOR:ACT:제14조",
                    "CHILD_ENUMERATION:ACT:제14조:1호",
                ],
            }
        ],
    )
    assert training_gate(rows)["status"] == "BLOCKED"
    assert rows[0]["parent_example_id"] == "Q1"
    assert rows[0]["pattern_family"] == "CONDITION_EXCEPTION"
    assert rows[0]["lineage_pattern_family"] == "CONDITION_EXCEPTION"
    assert rows[0]["ranking_training_eligible"] is True
    result = recommend_knn("직접지급 요건은?", rows, k=1)
    assert result["status"] == "EXPERIMENTAL_RECOMMENDATION"
    assert result["retrieval_actions"] == [
        "SEARCH_ANCHOR",
        "EXPAND_CHILDREN",
    ]


def test_intent_changing_variant_is_excluded_from_ranking() -> None:
    qa = {
        "reference_qa_id": "Q2",
        "customer_id": "C",
        "reference_version": "v1",
        "source_kind": "SYNTHETIC",
        "question": "예외도 포함해서 지급기한을 알려주세요.",
        "metadata": {
            "intent": "deadline",
            "variant_id": "exception_check",
        },
    }
    rows, _ = compile_training_rows(
        [qa],
        [{"reference_qa_id": "Q2", "answer_flow": ["CONDITION"]}],
        [
            {
                "reference_qa_id": "Q2",
                "traversal_flow": ["ANCHOR:ACT:제13조"],
            }
        ],
    )

    assert rows[0]["ranking_training_eligible"] is False
    assert (
        rows[0]["ranking_exclusion_reason"]
        == "QUESTION_INTENT_CHANGED_WITHOUT_ANSWER_REGENERATION"
    )
