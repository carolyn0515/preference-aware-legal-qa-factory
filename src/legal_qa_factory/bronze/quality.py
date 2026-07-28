from __future__ import annotations

from typing import Any

from legal_qa_factory.common.hashing import sha256_text


def validate_records(records: list[dict[str, Any]], expected_hash: str) -> None:
    if not records:
        raise ValueError("Bronze dataset is empty")
    ids: set[str] = set()
    natural_keys: set[tuple[str, int, int]] = set()
    for record in records:
        record_id = record["bronze_record_id"]
        natural_key = (
            record["raw_object_id"],
            record["page_number"],
            record["block_index"],
        )
        if record_id in ids:
            raise ValueError(f"duplicate bronze_record_id: {record_id}")
        if natural_key in natural_keys:
            raise ValueError(f"duplicate natural key: {natural_key}")
        if record["content_hash"] != expected_hash:
            raise ValueError(f"source hash mismatch: {record_id}")
        if record["text_sha256"] != sha256_text(record["normalized_text"]):
            raise ValueError(f"text hash mismatch: {record_id}")
        if record["character_count"] != len(record["normalized_text"]):
            raise ValueError(f"character count mismatch: {record_id}")
        bbox = record["bounding_box"]
        invalid_coordinates = (
            bbox["x1"] < bbox["x0"] or bbox["y1"] < bbox["y0"]
        )
        if record["page_number"] < 1 or invalid_coordinates:
            raise ValueError(f"invalid physical coordinates: {record_id}")
        ids.add(record_id)
        natural_keys.add(natural_key)
