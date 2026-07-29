from __future__ import annotations

from typing import Any

from legal_qa_factory.common.hashing import sha256_text


def validate_records(records: list[dict[str, Any]], expected_hash: str) -> None:
    if not records:
        raise ValueError("Bronze dataset is empty")
    ids = set()
    keys = set()
    for record in records:
        record_id = record["bronze_record_id"]
        key = (record["raw_object_id"], record["page_number"], record["block_index"])
        if record_id in ids or key in keys:
            raise ValueError(f"duplicate Bronze identity: {record_id}")
        if record["content_hash"] != expected_hash:
            raise ValueError(f"source hash mismatch: {record_id}")
        if record["text_sha256"] != sha256_text(record["normalized_text"]):
            raise ValueError(f"text hash mismatch: {record_id}")
        if record["character_count"] != len(record["normalized_text"]):
            raise ValueError(f"character count mismatch: {record_id}")
        if not record["lines"]:
            raise ValueError(f"missing typography lines: {record_id}")
        spans = [span for line in record["lines"] for span in line["spans"]]
        if not spans or any(span["font_size"] <= 0 for span in spans):
            raise ValueError(f"invalid typography spans: {record_id}")
        if not record["typography_traces"]:
            raise ValueError(f"missing typography traces: {record_id}")
        ids.add(record_id)
        keys.add(key)
