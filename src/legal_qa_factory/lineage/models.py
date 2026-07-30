from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa


@dataclass(frozen=True)
class EvidenceLink:
    claim_id: str
    legal_node_id: str
    lineage_kind: str  # OBSERVED or INFERRED
    inference_method: str | None
    confidence: float | None
    rank: int


CLAIM_FEATURE_SCHEMA = pa.schema(
    [
        pa.field("reference_claim_id", pa.string(), nullable=False),
        pa.field("reference_qa_id", pa.string(), nullable=False),
        pa.field("claim_sequence", pa.int32(), nullable=False),
        pa.field("text", pa.large_string(), nullable=False),
        pa.field("keywords", pa.list_(pa.string()), nullable=False),
        pa.field("answer_roles", pa.list_(pa.string()), nullable=False),
        pa.field("explicit_citations", pa.list_(pa.string()), nullable=False),
        pa.field("has_negation", pa.bool_(), nullable=False),
    ]
)

CLAIM_EVIDENCE_SCHEMA = pa.schema(
    [
        pa.field("reference_claim_id", pa.string(), nullable=False),
        pa.field("reference_qa_id", pa.string(), nullable=False),
        pa.field("claim_sequence", pa.int32(), nullable=False),
        pa.field("evidence_proposition_id", pa.string(), nullable=False),
        pa.field("evidence_legal_node_id", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("citation_label", pa.string(), nullable=False),
        pa.field("article_citation_label", pa.string(), nullable=False),
        pa.field("article_title", pa.string()),
        pa.field("node_type", pa.string(), nullable=False),
        pa.field("evidence_text", pa.large_string(), nullable=False),
        pa.field("lineage_kind", pa.string(), nullable=False),
        pa.field("retrieval_relation", pa.string(), nullable=False),
        pa.field("retrieval_path", pa.list_(pa.string()), nullable=False),
        pa.field("matched_terms", pa.list_(pa.string()), nullable=False),
        pa.field("bm25_score", pa.float32(), nullable=False),
        pa.field("query_coverage", pa.float32(), nullable=False),
        pa.field("title_overlap", pa.float32(), nullable=False),
        pa.field("citation_score", pa.float32(), nullable=False),
        pa.field("final_score", pa.float32(), nullable=False),
        pa.field("rank", pa.int32(), nullable=False),
        pa.field("selected", pa.bool_(), nullable=False),
        pa.field("selection_status", pa.string(), nullable=False),
        pa.field("legal_functions", pa.list_(pa.string()), nullable=False),
        pa.field("legal_function_usable", pa.bool_(), nullable=False),
    ]
)

QA_FLOW_SCHEMA = pa.schema(
    [
        pa.field("reference_qa_id", pa.string(), nullable=False),
        pa.field("answer_flow", pa.list_(pa.string()), nullable=False),
        pa.field("retrieval_flow", pa.list_(pa.string()), nullable=False),
        pa.field("evidence_citations", pa.list_(pa.string()), nullable=False),
        pa.field("grounded_claim_count", pa.int32(), nullable=False),
        pa.field("claim_count", pa.int32(), nullable=False),
        pa.field("candidate_grounding_rate", pa.float32(), nullable=False),
    ]
)
