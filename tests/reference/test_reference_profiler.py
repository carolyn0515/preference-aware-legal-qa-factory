from legal_qa_factory.reference.profiler import (
    aggregate_profile,
    observed_features,
)


def test_observed_features_are_structural_not_inferred() -> None:
    qa = {
        "reference_qa_id": "RQA-1",
        "customer_id": "CUSTOMER",
        "reference_version": "v1",
        "question": "가능한가요?",
        "answer": "할 수 있습니다. 다만 사실관계를 확인해야 합니다.",
    }
    claims = [
        {"text": "할 수 있습니다."},
        {"text": "다만 사실관계를 확인해야 합니다."},
    ]
    result = observed_features(qa, claims)
    assert result["has_conclusion_marker"] is True
    assert result["first_claim_has_conclusion"] is True
    assert result["has_exception_marker"] is True
    assert result["has_uncertainty_marker"] is True


def test_small_sample_does_not_become_confirmed_preference() -> None:
    feature = {
        "reference_qa_id": "RQA-1",
        "customer_id": "CUSTOMER",
        "reference_version": "v1",
        "question_char_count": 5,
        "answer_char_count": 20,
        "claim_count": 1,
        "paragraph_count": 1,
        "list_item_count": 0,
        "article_citation_count": 0,
        "law_title_count": 0,
        "has_condition_marker": True,
        "has_exception_marker": False,
        "has_procedure_marker": False,
        "has_uncertainty_marker": False,
        "has_conclusion_marker": True,
        "first_claim_has_conclusion": True,
    }
    profile = aggregate_profile([feature], input_sha256="abc")
    assert {
        item["status"] for item in profile["preference_candidates"]
    } == {"INSUFFICIENT_SAMPLE"}
