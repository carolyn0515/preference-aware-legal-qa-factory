from legal_qa_factory.review.exporter import system_hash
from legal_qa_factory.review.importer import (
    apply_reviews,
    validate_review_rows,
)


def reviewed_row(system: dict, decision: str, review_fields: set[str]) -> dict:
    row = {
        "review_row_id": system.pop("review_row_id"),
        **system,
        "human_decision": decision,
        "reviewer": "reviewer",
        "comment": "",
    }
    row["system_row_sha256"] = system_hash(row, review_fields)
    return row


def test_modified_system_column_is_rejected() -> None:
    fields = {"human_decision", "reviewer", "comment"}
    row = reviewed_row(
        {
            "review_row_id": "R1",
            "reference_qa_id": "Q1",
            "evidence_text": "original",
        },
        "CORRECT",
        fields,
    )
    row["evidence_text"] = "modified"
    try:
        validate_review_rows(
            [row],
            allowed_decisions=frozenset({"CORRECT"}),
            review_fields=fields,
        )
    except ValueError as error:
        assert "system fields were modified" in str(error)
    else:
        raise AssertionError("modified system fields must fail validation")


def test_reviewed_synthetic_row_stays_production_ineligible() -> None:
    answer_fields = {
        "human_decision",
        "corrected_roles",
        "reviewer",
        "comment",
    }
    evidence_fields = {"human_decision", "reviewer", "comment"}
    retrieval_fields = {"human_decision", "reviewer", "comment"}
    answer = reviewed_row(
        {
            "review_row_id": "A1",
            "reference_qa_id": "Q1",
            "reference_claim_id": "C1",
            "claim_sequence": "1",
            "system_roles": "CONCLUSION",
            "corrected_roles": "",
        },
        "CORRECT",
        answer_fields,
    )
    evidence = reviewed_row(
        {
            "review_row_id": "E1",
            "reference_qa_id": "Q1",
            "reference_claim_id": "C1",
        },
        "CORRECT",
        evidence_fields,
    )
    retrieval = reviewed_row(
        {
            "review_row_id": "T1",
            "reference_qa_id": "Q1",
            "reference_claim_id": "C1",
            "sequence": "1",
            "system_action": "SEARCH_ANCHOR",
        },
        "REQUIRED",
        retrieval_fields,
    )
    policy = {
        "reference_qa_id": "Q1",
        "source_kind": "SYNTHETIC",
        "answer_flow": ["CONCLUSION"],
        "retrieval_actions": ["SEARCH_ANCHOR"],
    }
    result = apply_reviews([policy], [answer], [evidence], [retrieval])
    assert result[0]["label_provenance"] == "HUMAN_REVIEWED"
    assert result[0]["production_training_eligible"] is False
    assert result[0]["sample_weight"] == 0.4
