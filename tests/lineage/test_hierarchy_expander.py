from legal_qa_factory.lineage.hierarchy_expander import expand_anchor


def test_selected_enumeration_lead_expands_to_children() -> None:
    article = {
        "legal_node_id": "A1",
        "article_node_id": "A1",
        "parent_node_id": None,
        "source_id": "ACT",
        "source_type": "STATUTE",
        "node_type": "ARTICLE",
        "citation_label": "제14조",
        "text": "",
    }
    paragraph = {
        "legal_node_id": "N1",
        "article_node_id": "A1",
        "parent_node_id": "A1",
        "source_id": "ACT",
        "source_type": "STATUTE",
        "node_type": "PARAGRAPH",
        "citation_label": "①",
        "text": "다음 각 호의 사유가 발생한 때에는 직접 지급한다.",
    }
    item = {
        "legal_node_id": "N2",
        "article_node_id": "A1",
        "parent_node_id": "N1",
        "source_id": "ACT",
        "source_type": "STATUTE",
        "node_type": "ITEM",
        "citation_label": "1호",
        "text": "원사업자가 지급할 수 없는 경우",
    }
    anchor = {
        "reference_claim_id": "C1",
        "reference_qa_id": "Q1",
        "evidence_proposition_id": "P1",
        "evidence_legal_node_id": "N1",
        "source_id": "ACT",
        "final_score": 0.8,
    }
    result = expand_anchor(
        anchor=anchor,
        answer_roles=["CONDITION"],
        propositions_by_node={
            "N1": [{"proposition_id": "P1", "source_id": "ACT", "text": "lead"}],
            "N2": [{"proposition_id": "P2", "source_id": "ACT", "text": "item"}],
        },
        nodes_by_id={"A1": article, "N1": paragraph, "N2": item},
        children_by_node={"A1": [paragraph], "N1": [item]},
        article_nodes={("ACT", "제14조"): article},
        decree_reference_index={},
    )
    assert result[0]["context_proposition_id"] == "P2"
    assert result[0]["expansion_relation"] == "CHILD_ENUMERATION"
