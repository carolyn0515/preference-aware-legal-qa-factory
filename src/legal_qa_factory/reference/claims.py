from __future__ import annotations

import re
from typing import Any

from legal_qa_factory.common.hashing import sha256_text

BOUNDARY = re.compile(r"(?<=[.!?。])(?:\s+|\n+)|\n+(?=[^\s])")


def split_answer_claims(
    reference_qa_id: str, answer: str
) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    cursor = 0
    for match in BOUNDARY.finditer(answer):
        _append_claim(claims, reference_qa_id, answer, cursor, match.start())
        cursor = match.end()
    _append_claim(claims, reference_qa_id, answer, cursor, len(answer))
    return claims


def _append_claim(
    claims: list[dict[str, Any]],
    reference_qa_id: str,
    answer: str,
    start: int,
    end: int,
) -> None:
    raw = answer[start:end]
    stripped = raw.strip()
    if not stripped:
        return
    leading = len(raw) - len(raw.lstrip())
    trailing = len(raw.rstrip())
    exact_start = start + leading
    exact_end = start + trailing
    sequence = len(claims) + 1
    claims.append(
        {
            "reference_claim_id": "RCL-"
            + sha256_text(f"{reference_qa_id}:{sequence}:{stripped}"),
            "reference_qa_id": reference_qa_id,
            "claim_sequence": sequence,
            "text": stripped,
            "char_start": exact_start,
            "char_end": exact_end,
            "split_rule": "PUNCTUATION_OR_NEWLINE_V1",
        }
    )
