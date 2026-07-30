from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from legal_qa_factory.common.hashing import sha256_text
from legal_qa_factory.reference.validator import (
    normalized_question,
    validate_input_rows,
    validate_single_partition,
)


@dataclass(frozen=True)
class ExpansionResult:
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number}: expected JSON object")
            rows.append(value)
    return rows


def read_plan(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        plan = yaml.safe_load(stream)
    if not isinstance(plan, dict):
        raise ValueError("generation plan must be a YAML object")
    return plan


def seed_identity(row: dict[str, Any]) -> str:
    canonical = "|".join(
        [
            row["customer_id"],
            row["reference_version"],
            normalized_question(row["question"]),
            row["answer"].strip(),
        ]
    )
    return "SEED-" + sha256_text(canonical)[:24]


def _metadata(
    seed: dict[str, Any],
    *,
    parent_id: str,
    variant_id: str,
    question_type: str,
    difficulty: str,
    method: str,
) -> dict[str, Any]:
    metadata = deepcopy(seed.get("metadata", {}))
    metadata.update(
        {
            "parent_example_id": parent_id,
            "variant_id": variant_id,
            "question_type": question_type,
            "difficulty": difficulty,
            "generation_method": method,
        }
    )
    return metadata


def expand_rows(
    seeds: list[dict[str, Any]], plan: dict[str, Any]
) -> ExpansionResult:
    dataset = plan["dataset"]
    generation = plan["generation"]
    variants = generation["variants"]
    method = generation["method"]
    expected_seed_count = plan["seed"]["expected_count"]
    if len(seeds) != expected_seed_count:
        raise ValueError(
            f"seed count mismatch: expected={expected_seed_count} "
            f"actual={len(seeds)}"
        )
    if len(variants) != generation["variants_per_seed"]:
        raise ValueError("variants_per_seed does not match variants")

    output: list[dict[str, Any]] = []
    for seed in seeds:
        parent_id = seed_identity(seed)
        common = {
            "customer_id": dataset["customer_id"],
            "reference_version": dataset["reference_version"],
            "source_kind": dataset["source_kind"],
            "answer": seed["answer"].strip(),
            "observed_evidence_ids": list(
                seed.get("observed_evidence_ids", [])
            ),
        }
        if generation["include_seed_form"]:
            output.append(
                {
                    **common,
                    "question": seed["question"].strip(),
                    "metadata": _metadata(
                        seed,
                        parent_id=parent_id,
                        variant_id="base",
                        question_type="BASE",
                        difficulty="BASIC",
                        method=method,
                    ),
                }
            )
        for variant in variants:
            output.append(
                {
                    **common,
                    "question": (
                        variant["prefix"].strip()
                        + " "
                        + seed["question"].strip()
                    ),
                    "metadata": _metadata(
                        seed,
                        parent_id=parent_id,
                        variant_id=variant["id"],
                        question_type=variant["question_type"],
                        difficulty=variant["difficulty"],
                        method=method,
                    ),
                }
            )

    validate_expansion(output, plan)
    type_counts = Counter(
        row["metadata"]["question_type"] for row in output
    )
    difficulty_counts = Counter(
        row["metadata"]["difficulty"] for row in output
    )
    topic_counts = Counter(
        row["metadata"].get("topic", "UNKNOWN") for row in output
    )
    return ExpansionResult(
        rows=output,
        manifest={
            "schema_version": "1.0",
            "reference_version": dataset["reference_version"],
            "source_kind": dataset["source_kind"],
            "generation_method": method,
            "seed_count": len(seeds),
            "row_count": len(output),
            "parent_group_count": len(
                {row["metadata"]["parent_example_id"] for row in output}
            ),
            "question_type_counts": dict(sorted(type_counts.items())),
            "difficulty_counts": dict(sorted(difficulty_counts.items())),
            "topic_counts": dict(sorted(topic_counts.items())),
            "split_policy": "GROUP_BY_PARENT_EXAMPLE_ID",
        },
    )


def validate_expansion(
    rows: list[dict[str, Any]], plan: dict[str, Any]
) -> None:
    quality = plan["quality"]
    errors: list[str] = []
    target_count = plan["dataset"]["target_count"]
    if len(rows) != target_count:
        errors.append(
            f"target count mismatch: expected={target_count} actual={len(rows)}"
        )
    allowed_types = set(quality["allowed_question_types"])
    allowed_difficulties = set(quality["allowed_difficulties"])
    for index, row in enumerate(rows, start=1):
        metadata = row.get("metadata", {})
        if (
            quality["require_evidence"]
            and not row.get("observed_evidence_ids")
        ):
            errors.append(f"row {index}: evidence is required")
        if (
            quality["require_parent_example_id"]
            and not metadata.get("parent_example_id")
        ):
            errors.append(f"row {index}: parent_example_id is required")
        if metadata.get("question_type") not in allowed_types:
            errors.append(f"row {index}: invalid question_type")
        if metadata.get("difficulty") not in allowed_difficulties:
            errors.append(f"row {index}: invalid difficulty")
    if errors:
        raise ValueError("invalid expansion:\n- " + "\n- ".join(errors))
    validate_input_rows(rows)
    validate_single_partition(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    pending.replace(path)

