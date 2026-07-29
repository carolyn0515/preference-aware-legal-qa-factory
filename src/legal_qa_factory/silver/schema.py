from __future__ import annotations

import pyarrow as pa

SILVER_SCHEMA = pa.schema(
    [
        pa.field("legal_node_id", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_type", pa.string(), nullable=False),
        pa.field("source_version_hash", pa.string(), nullable=False),
        pa.field("node_type", pa.string(), nullable=False),
        pa.field("parent_node_id", pa.string()),
        pa.field("article_node_id", pa.string(), nullable=False),
        pa.field("sequence", pa.int32(), nullable=False),
        pa.field("citation_label", pa.string(), nullable=False),
        pa.field("title", pa.string()),
        pa.field("marker", pa.string(), nullable=False),
        pa.field("text", pa.large_string(), nullable=False),
        pa.field("region", pa.string(), nullable=False),
        pa.field("page_from", pa.int32(), nullable=False),
        pa.field("page_to", pa.int32(), nullable=False),
        pa.field("bronze_record_ids", pa.list_(pa.string()), nullable=False),
        pa.field("boundary_confidence", pa.float32(), nullable=False),
        pa.field("boundary_evidence", pa.list_(pa.string()), nullable=False),
        pa.field("parser_id", pa.string(), nullable=False),
        pa.field("transaction_from", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("transaction_to", pa.timestamp("us", tz="UTC")),
    ],
    metadata={b"schema_name": b"silver_legal_node", b"schema_version": b"1.0"},
)
