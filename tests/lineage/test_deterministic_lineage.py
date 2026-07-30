from legal_qa_factory.lineage.evidence_alignment import align_claim
from legal_qa_factory.lineage.trace_builder import (
    aggregate_flow_patterns,
    build_qa_flows,
)
from legal_qa_factory.retrieval.lexical import BM25Index


def test_alignment_finds_lexically_related_evidence() -> None:
    node = {
        "legal_node_id": "N1",
        "article_node_id": "N1",
        "parent_node_id": None,
        "node_type": "ARTICLE",
        "citation_label": "제1조",
        "title": "대금 지급",
    }
    proposition = {
        "proposition_id": "P1",
        "legal_node_id": "N1",
        "source_id": "ACT",
        "text": "원사업자는 하도급대금을 지급하여야 한다.",
        "retrieval_text": "제1조 대금 지급 원사업자는 하도급대금을 지급하여야 한다.",
    }
    claim = {
        "reference_claim_id": "C1",
        "reference_qa_id": "Q1",
        "claim_sequence": 1,
        "text": "원사업자는 하도급대금을 지급해야 합니다.",
    }
    feature, candidates = align_claim(
        claim=claim,
        index=BM25Index([proposition]),
        nodes_by_id={"N1": node},
        functions_by_proposition={},
        legal_function_usable=False,
        top_k=1,
    )
    assert feature["answer_roles"] == ["CONCLUSION"]
    assert candidates[0]["evidence_proposition_id"] == "P1"
    assert candidates[0]["selected"] is True


def test_flow_marks_claim_without_selected_evidence() -> None:
    features = [
        {
            "reference_claim_id": "C1",
            "reference_qa_id": "Q1",
            "claim_sequence": 1,
            "answer_roles": ["PRACTICAL_GUIDANCE"],
        }
    ]
    flows = build_qa_flows(features, [])
    assert flows[0]["retrieval_flow"] == ["NO_DIRECT_LEGAL_EVIDENCE"]
    assert flows[0]["candidate_grounding_rate"] == 0
    patterns = aggregate_flow_patterns(flows)
    assert (
        patterns["answer_flow_patterns"][0]["status"]
        == "INSUFFICIENT_SAMPLE"
    )
