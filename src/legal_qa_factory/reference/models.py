from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyarrow as pa


@dataclass(frozen=True)
class ReferenceQA:
    reference_qa_id: str
    question: str
    answer: str
    customer_id: str
    reference_version: str
    source_kind: str
    source_row_number: int
    observed_evidence_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None


REFERENCE_QA_SCHEMA = pa.schema(
    [
        pa.field("reference_qa_id", pa.string(), nullable=False),
        pa.field("question", pa.large_string(), nullable=False),
        pa.field("answer", pa.large_string(), nullable=False),
        pa.field("customer_id", pa.string(), nullable=False),
        pa.field("reference_version", pa.string(), nullable=False),
        pa.field("source_kind", pa.string(), nullable=False),
        pa.field("source_row_number", pa.int32(), nullable=False),
        pa.field("observed_evidence_ids", pa.list_(pa.string()), nullable=False),
        pa.field("metadata_json", pa.large_string(), nullable=False),
        pa.field("input_sha256", pa.string(), nullable=False),
    ],
    metadata={
        b"schema_name": b"reference_qa",
        b"schema_version": b"1.0",
        b"truth_semantics": b"PREFERENCE_EVIDENCE_ONLY",
    },
)

REFERENCE_CLAIM_SCHEMA = pa.schema(
    [
        pa.field("reference_claim_id", pa.string(), nullable=False),
        pa.field("reference_qa_id", pa.string(), nullable=False),
        pa.field("claim_sequence", pa.int32(), nullable=False),
        pa.field("text", pa.large_string(), nullable=False),
        pa.field("char_start", pa.int32(), nullable=False),
        pa.field("char_end", pa.int32(), nullable=False),
        pa.field("split_rule", pa.string(), nullable=False),
    ],
    metadata={b"schema_name": b"reference_claim", b"schema_version": b"1.0"},
)
