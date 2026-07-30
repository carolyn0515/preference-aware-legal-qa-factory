from __future__ import annotations

import re
from collections import Counter
from typing import Any

ALLOWED_SOURCE_KINDS = frozenset({"CUSTOMER_GOLD", "SYNTHETIC"})
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def normalized_question(value: str) -> str:
    return " ".join(value.split()).casefold()


def validate_input_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("reference dataset must contain at least one QA row")
    errors: list[str] = []
    normalized_questions: list[str] = []
    for index, row in enumerate(rows, start=1):
        for field in ("question", "answer", "customer_id", "reference_version"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"row {index}: {field} must be a non-empty string")
        source_kind = row.get("source_kind")
        if source_kind not in ALLOWED_SOURCE_KINDS:
            errors.append(
                f"row {index}: source_kind must be one of "
                f"{sorted(ALLOWED_SOURCE_KINDS)}"
            )
        for field in ("customer_id", "reference_version"):
            value = row.get(field)
            if isinstance(value, str) and value and not IDENTIFIER.fullmatch(value):
                errors.append(f"row {index}: invalid {field}: {value!r}")
        evidence = row.get("observed_evidence_ids", [])
        if not isinstance(evidence, list) or not all(
            isinstance(value, str) and value.strip() for value in evidence
        ):
            errors.append(
                f"row {index}: observed_evidence_ids must be a string list"
            )
        if not isinstance(row.get("metadata", {}), dict):
            errors.append(f"row {index}: metadata must be an object")
        question = row.get("question")
        if isinstance(question, str) and question.strip():
            normalized_questions.append(normalized_question(question))

    duplicates = [
        question
        for question, count in Counter(normalized_questions).items()
        if count > 1
    ]
    if duplicates:
        errors.append(f"duplicate normalized questions: {duplicates[:3]}")
    if errors:
        raise ValueError("invalid reference dataset:\n- " + "\n- ".join(errors))


def validate_single_partition(rows: list[dict[str, Any]]) -> tuple[str, str]:
    customers = {row["customer_id"] for row in rows}
    versions = {row["reference_version"] for row in rows}
    if len(customers) != 1 or len(versions) != 1:
        raise ValueError(
            "one input file must contain exactly one customer_id and "
            "one reference_version"
        )
    return next(iter(customers)), next(iter(versions))
