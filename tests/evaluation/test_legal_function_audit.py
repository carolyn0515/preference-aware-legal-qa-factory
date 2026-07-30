from legal_qa_factory.evaluation.legal_function_audit import (
    stratified_sample,
    suspicious_reasons,
)


def row(identifier: str, source: str, node_type: str, length: str) -> dict:
    return {
        "proposition_id": identifier,
        "source_id": source,
        "node_type": node_type,
        "length_bucket": length,
    }


def test_stratified_sample_is_deterministic_and_covers_strata() -> None:
    rows = [
        row("p1", "ACT", "ARTICLE", "SHORT"),
        row("p2", "ACT", "ARTICLE", "SHORT"),
        row("p3", "ACT", "PARAGRAPH", "MEDIUM"),
        row("p4", "DECREE", "ARTICLE", "SHORT"),
    ]
    first = stratified_sample(rows, 3)
    second = stratified_sample(rows, 3)
    assert [item["proposition_id"] for item in first] == [
        item["proposition_id"] for item in second
    ]
    assert len({(item["source_id"], item["node_type"]) for item in first}) == 3


def test_suspicious_reasons_identifies_invalid_combinations() -> None:
    reasons = suspicious_reasons(
        {
            "classified": True,
            "intrinsic_labels": ["UNCLASSIFIED", "OBLIGATION"],
            "semantic_unit_type": "FULL_PROPOSITION",
            "action": None,
            "evidence_phrases": [],
            "confidence": 0.5,
        }
    )
    assert "UNCLASSIFIED_WITH_OTHER_LABEL" in reasons
    assert "OBLIGATION_WITHOUT_ACTION" in reasons
    assert "MISSING_EVIDENCE" in reasons
    assert "LOW_CONFIDENCE" in reasons
