from __future__ import annotations

import pyarrow as pa

POLICY_TRAINING_SCHEMA = pa.schema(
    [
        pa.field("reference_qa_id", pa.string(), nullable=False),
        pa.field("parent_example_id", pa.string(), nullable=False),
        pa.field("customer_id", pa.string(), nullable=False),
        pa.field("reference_version", pa.string(), nullable=False),
        pa.field("source_kind", pa.string(), nullable=False),
        pa.field("question", pa.large_string(), nullable=False),
        pa.field("question_terms", pa.list_(pa.string()), nullable=False),
        pa.field("question_intents", pa.list_(pa.string()), nullable=False),
        pa.field("has_explicit_citation", pa.bool_(), nullable=False),
        pa.field("asks_condition", pa.bool_(), nullable=False),
        pa.field("asks_exception", pa.bool_(), nullable=False),
        pa.field("asks_procedure", pa.bool_(), nullable=False),
        pa.field("asks_deadline", pa.bool_(), nullable=False),
        pa.field("asks_sanction", pa.bool_(), nullable=False),
        pa.field("pattern_id", pa.string(), nullable=False),
        pa.field("pattern_family", pa.string(), nullable=False),
        pa.field("family_label_source", pa.string(), nullable=False),
        pa.field("answer_flow", pa.list_(pa.string()), nullable=False),
        pa.field("retrieval_actions", pa.list_(pa.string()), nullable=False),
        pa.field("requires_decree", pa.bool_(), nullable=False),
        pa.field("requires_exception_search", pa.bool_(), nullable=False),
        pa.field("requires_child_expansion", pa.bool_(), nullable=False),
        pa.field("label_provenance", pa.string(), nullable=False),
        pa.field("sample_weight", pa.float32(), nullable=False),
        pa.field("production_training_eligible", pa.bool_(), nullable=False),
    ],
    metadata={
        b"schema_name": b"policy_training_example",
        b"schema_version": b"1.2",
    },
)
