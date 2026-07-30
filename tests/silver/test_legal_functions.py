import pytest

from legal_qa_factory.silver.semantics.legal_functions import (
    align_evidence_phrase,
    apply_label_inheritance,
    build_payload,
    request_hash,
    validate_result,
)

PROPOSITION = {
    "proposition_id": "PRP-1",
    "legal_node_id": "NODE-1",
    "text": "원사업자는 서면을 발급하여야 한다.",
}
NODE = {
    "legal_node_id": "NODE-1",
    "article_node_id": "ARTICLE-1",
    "source_type": "STATUTE",
    "citation_label": "①",
    "title": None,
    "node_type": "PARAGRAPH",
}


def test_request_hash_is_deterministic() -> None:
    payload = build_payload([PROPOSITION], {"NODE-1": NODE})
    first = request_hash(namespace="v1", model="test", prompt_id="p1", payload=payload)
    second = request_hash(namespace="v1", model="test", prompt_id="p1", payload=payload)
    assert first == second


def test_evidence_must_exist_in_source() -> None:
    response = {
        "results": [
            {
                "proposition_id": "PRP-1",
                "labels": ["OBLIGATION"],
                "evidence_phrases": ["존재하지 않는 문구"],
            }
        ]
    }
    with pytest.raises(ValueError, match="evidence phrases"):
        validate_result(response, [PROPOSITION])


def test_duplicate_labels_are_rejected_locally() -> None:
    response = {
        "results": [
            {
                "proposition_id": "PRP-1",
                "labels": ["OBLIGATION", "OBLIGATION"],
                "evidence_phrases": ["발급하여야 한다"],
            }
        ]
    }
    with pytest.raises(ValueError, match="duplicate labels"):
        validate_result(response, [PROPOSITION])


def test_evidence_is_realigned_to_pdf_source_spacing() -> None:
    source = "그 시행에 필요한 사항을 규정함을 목적으 로 한다."
    assert (
        align_evidence_phrase(source, "필요한 사항을 규정함을 목적으로 한다")
        == "필요한 사항을 규정함을 목적으 로 한다"
    )


def test_validation_stores_exact_source_evidence() -> None:
    proposition = {
        **PROPOSITION,
        "text": "수 급사업자는 서면을 발급하여야 한다.",
    }
    response = {
        "results": [
            {
                "proposition_id": "PRP-1",
                "labels": ["OBLIGATION"],
                "evidence_phrases": ["수급사업자는 서면을 발급하여야 한다"],
            }
        ]
    }
    validate_result(response, [proposition])
    assert response["results"][0]["evidence_phrases"] == [
        "수 급사업자는 서면을 발급하여야 한다"
    ]


def test_unclassified_is_allowed_as_exclusive_fallback() -> None:
    response = {
        "results": [
            {
                "proposition_id": "PRP-1",
                "labels": ["UNCLASSIFIED"],
                "evidence_phrases": ["원사업자는"],
            }
        ]
    }
    validate_result(response, [PROPOSITION])


def test_unclassified_cannot_coexist_with_legal_function() -> None:
    response = {
        "results": [
            {
                "proposition_id": "PRP-1",
                "labels": ["UNCLASSIFIED", "OBLIGATION"],
                "evidence_phrases": ["발급하여야 한다"],
            }
        ]
    }
    with pytest.raises(ValueError, match="must be exclusive"):
        validate_result(response, [PROPOSITION])


def test_payload_contains_article_and_parent_context() -> None:
    article = {
        **NODE,
        "legal_node_id": "ARTICLE-1",
        "article_node_id": "ARTICLE-1",
        "parent_node_id": None,
        "node_type": "ARTICLE",
        "citation_label": "제2조",
        "title": "정의",
        "text": "",
    }
    paragraph = {
        **NODE,
        "parent_node_id": "ARTICLE-1",
        "text": "다음 각 호의 어느 하나에 해당하는 자를 말한다.",
    }
    proposition = {**PROPOSITION, "legal_node_id": "NODE-1"}
    payload = build_payload(
        [proposition],
        {"ARTICLE-1": article, "NODE-1": paragraph},
    )
    item = payload["propositions"][0]
    assert item["article_title"] == "정의"
    assert item["ancestor_path"] == ["제2조", "①"]
    assert item["parent_lead_in"] == ""


def test_list_fragment_inherits_definition_from_parent_lead() -> None:
    article = {
        **NODE,
        "legal_node_id": "ARTICLE-1",
        "article_node_id": "ARTICLE-1",
        "parent_node_id": None,
        "node_type": "ARTICLE",
        "citation_label": "제2조",
        "title": "정의",
        "text": "",
    }
    parent = {
        **NODE,
        "parent_node_id": "ARTICLE-1",
        "text": "다음 각 호의 어느 하나에 해당하는 자를 말한다.",
    }
    child = {
        **NODE,
        "legal_node_id": "ITEM-1",
        "parent_node_id": "NODE-1",
        "node_type": "ITEM",
        "citation_label": "1호",
        "text": "물품의 제조",
    }
    propositions = {
        "LEAD": {
            **PROPOSITION,
            "proposition_id": "LEAD",
            "legal_node_id": "NODE-1",
            "text": parent["text"],
        },
        "FRAGMENT": {
            **PROPOSITION,
            "proposition_id": "FRAGMENT",
            "legal_node_id": "ITEM-1",
            "text": child["text"],
        },
    }
    records = [
        {"proposition_id": "LEAD", "labels": ["DEFINITION"]},
        {"proposition_id": "FRAGMENT", "labels": ["UNCLASSIFIED"]},
    ]
    result = apply_label_inheritance(
        records,
        propositions,
        {"ARTICLE-1": article, "NODE-1": parent, "ITEM-1": child},
    )
    fragment = result[1]
    assert fragment["intrinsic_labels"] == ["UNCLASSIFIED"]
    assert fragment["inherited_labels"] == ["DEFINITION"]
    assert fragment["labels"] == ["DEFINITION"]
