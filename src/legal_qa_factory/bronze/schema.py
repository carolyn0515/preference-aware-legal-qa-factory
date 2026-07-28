from __future__ import annotations

import pyarrow as pa

BRONZE_SCHEMA = pa.schema(
    [
        pa.field("bronze_record_id", pa.string(), nullable=False),
        pa.field("raw_object_id", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_type", pa.string(), nullable=False),
        pa.field("content_hash", pa.string(), nullable=False),
        pa.field("page_number", pa.int32(), nullable=False),
        pa.field("block_index", pa.int32(), nullable=False),
        pa.field("parser_block_number", pa.int32(), nullable=False),
        pa.field(
            "bounding_box",
            pa.struct(
                [
                    pa.field("x0", pa.float32(), nullable=False),
                    pa.field("y0", pa.float32(), nullable=False),
                    pa.field("x1", pa.float32(), nullable=False),
                    pa.field("y1", pa.float32(), nullable=False),
                ]
            ),
            nullable=False,
        ),
        pa.field("raw_text", pa.large_string(), nullable=False),
        pa.field("normalized_text", pa.large_string(), nullable=False),
        pa.field("text_sha256", pa.string(), nullable=False),
        pa.field("character_count", pa.int32(), nullable=False),
        pa.field("parser_name", pa.string(), nullable=False),
        pa.field("parser_version", pa.string(), nullable=False),
        pa.field("extraction_run_id", pa.string(), nullable=False),
        pa.field("extracted_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ],
    metadata={b"schema_name": b"bronze_pdf_block", b"schema_version": b"1.0"},
)
