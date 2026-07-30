from legal_qa_factory.blueprints.evaluation import (
    benchmark_policy_models,
    classification_metrics,
)


def test_classification_metrics_match_known_counts() -> None:
    report = classification_metrics(
        ["A", "A", "B", "B"],
        ["A", "B", "B", "B"],
    )
    assert report["accuracy"] == 0.75
    assert report["per_class"]["A"]["precision"] == 1.0
    assert report["per_class"]["A"]["recall"] == 0.5
    assert report["per_class"]["B"]["recall"] == 1.0


def test_tiny_synthetic_data_does_not_select_best_model() -> None:
    rows = []
    for index, pattern in enumerate(("A", "B"), start=1):
        rows.append(
            {
                "reference_qa_id": f"Q{index}",
                "parent_example_id": f"G{index}",
                "question": f"질문 {index}",
                "question_terms": [f"질문{index}"],
                "question_intents": ["GENERAL_LEGAL_QA"],
                "has_explicit_citation": False,
                "asks_condition": False,
                "asks_exception": False,
                "asks_procedure": False,
                "asks_deadline": False,
                "asks_sanction": False,
                "pattern_id": pattern,
                "answer_flow": ["CONCLUSION"],
                "retrieval_actions": ["SEARCH_ANCHOR"],
                "sample_weight": 0.2,
                "production_training_eligible": False,
            }
        )
    result = benchmark_policy_models(rows)
    assert result["status"] == "NOT_REPORTABLE"
    assert result["best_model_id"] is None
    assert result["diagnostic_leader_id"] is not None


def test_evaluation_holds_out_all_sibling_variants() -> None:
    rows = []
    for group, pattern in (("G1", "A"), ("G2", "B")):
        for index in range(2):
            rows.append(
                {
                    "reference_qa_id": f"{group}-{index}",
                    "parent_example_id": group,
                    "question": f"{group} 질문 {index}",
                    "question_terms": [group, f"질문{index}"],
                    "question_intents": ["GENERAL_LEGAL_QA"],
                    "has_explicit_citation": False,
                    "asks_condition": False,
                    "asks_exception": False,
                    "asks_procedure": False,
                    "asks_deadline": False,
                    "asks_sanction": False,
                    "pattern_id": pattern,
                    "answer_flow": ["CONCLUSION"],
                    "retrieval_actions": ["SEARCH_ANCHOR"],
                    "sample_weight": 0.2,
                    "production_training_eligible": False,
                }
            )

    result = benchmark_policy_models(rows, k_values=(1,))

    model = result["models"][0]
    assert model["evaluation_strategy"] == "LEAVE_ONE_PARENT_GROUP_OUT"
    assert model["independent_group_count"] == 2
    assert result["reportability"]["physical_row_count"] == 4
    assert result["reportability"]["independent_group_count"] == 2
