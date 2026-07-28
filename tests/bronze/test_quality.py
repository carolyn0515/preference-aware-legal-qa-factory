import pytest

from legal_qa_factory.bronze.quality import validate_records
from legal_qa_factory.common.hashing import sha256_text


def valid_record() -> dict:
    text = "하도급법"
    return {
        "bronze_record_id": "BRZ-1",
        "raw_object_id": "RAW-1",
        "content_hash": "source-hash",
        "page_number": 1,
        "block_index": 0,
        "bounding_box": {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0},
        "normalized_text": text,
        "text_sha256": sha256_text(text),
        "character_count": len(text),
    }


def test_valid_record_passes() -> None:
    validate_records([valid_record()], "source-hash")


def test_text_hash_mismatch_blocks_publish() -> None:
    record = valid_record()
    record["text_sha256"] = "wrong"
    with pytest.raises(ValueError, match="text hash mismatch"):
        validate_records([record], "source-hash")
