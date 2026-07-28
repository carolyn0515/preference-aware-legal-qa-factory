from __future__ import annotations

from legal_qa_factory.common.hashing import sha256_text


def build_bronze_record_id(
    raw_object_id: str,
    page_number: int,
    block_index: int,
    text_sha256: str,
) -> str:
    identity = f"{raw_object_id}|{page_number}|{block_index}|{text_sha256}"
    return f"BRZ-{sha256_text(identity)}"
# record를 단순 row number가 아니라 원본 위치와 내용에 묶어 식별
