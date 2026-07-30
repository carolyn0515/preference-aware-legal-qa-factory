from __future__ import annotations

import pyarrow as pa

BLUEPRINT_RANKING_SCHEMA = pa.schema(
    [
        pa.field("reference_qa_id", pa.string(), nullable=False),
        pa.field("parent_example_id", pa.string(), nullable=False),
        pa.field("candidate_pattern_id", pa.string(), nullable=False),
        pa.field(
            "candidate_answer_flow",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field(
            "candidate_retrieval_actions",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field("relevance_score", pa.float32(), nullable=False),
        pa.field("relevance_grade", pa.int8(), nullable=False),
        pa.field("is_exact_pattern", pa.bool_(), nullable=False),
    ],
    metadata={
        b"schema_name": b"blueprint_ranking_example",
        b"schema_version": b"1.0",
        b"target_semantics": b"GRADED_RELEVANCE",
    },
)

