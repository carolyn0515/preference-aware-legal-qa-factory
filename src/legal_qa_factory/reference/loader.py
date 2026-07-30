from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from legal_qa_factory.common.hashing import sha256_file, sha256_text
from legal_qa_factory.reference.models import ReferenceQA
from legal_qa_factory.reference.validator import (
    validate_input_rows,
    validate_single_partition,
)


def load_jsonl(path: Path) -> tuple[list[ReferenceQA], str]:
    input_hash = sha256_file(path)
    raw_rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for row_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"row {row_number}: expected a JSON object")
            value["_source_row_number"] = row_number
            raw_rows.append(value)
    validate_input_rows(raw_rows)
    validate_single_partition(raw_rows)

    rows = []
    for value in raw_rows:
        identity = "|".join(
            [
                value["customer_id"],
                value["reference_version"],
                " ".join(value["question"].split()),
                value["answer"].strip(),
            ]
        )
        rows.append(
            ReferenceQA(
                reference_qa_id="RQA-" + sha256_text(identity),
                question=value["question"].strip(),
                answer=value["answer"].strip(),
                customer_id=value["customer_id"],
                reference_version=value["reference_version"],
                source_kind=value["source_kind"],
                source_row_number=value["_source_row_number"],
                observed_evidence_ids=tuple(
                    value.get("observed_evidence_ids", [])
                ),
                metadata=value.get("metadata", {}),
            )
        )
    return rows, input_hash
