from legal_qa_factory.blueprints.compiler import (
    abstract_retrieval_actions,
    compile_training_rows,
    question_features,
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
    result = recommend_knn("직접지급 요건은?", rows, k=1)
    assert result["status"] == "EXPERIMENTAL_RECOMMENDATION"
    assert result["retrieval_actions"] == [
        "SEARCH_ANCHOR",
        "EXPAND_CHILDREN",
    ]
