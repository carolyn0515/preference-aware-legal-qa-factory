from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from legal_qa_factory.blueprints.compiler import pattern_identity
from legal_qa_factory.review.exporter import system_hash

ANSWER_DECISIONS = frozenset({"CORRECT", "CHANGE", "REMOVE"})
EVIDENCE_DECISIONS = frozenset(
    {
        "CORRECT",
        "PARTIAL",
        "IRRELEVANT",
        "NO_EVIDENCE_APPROPRIATE",
        "MISSING_EVIDENCE",
    }
)
RETRIEVAL_DECISIONS = frozenset({"REQUIRED", "OPTIONAL", "UNNECESSARY"})


def read_review_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_review_rows(
    rows: list[dict[str, str]],
    *,
    allowed_decisions: frozenset[str],
    review_fields: set[str],
) -> None:
    if not rows:
        raise ValueError("review file must contain at least one row")
    seen = set()
    errors = []
    for number, row in enumerate(rows, start=2):
        row_id = row.get("review_row_id", "")
        if not row_id or row_id in seen:
            errors.append(f"line {number}: invalid or duplicate review_row_id")
        seen.add(row_id)
        if row.get("human_decision") not in allowed_decisions:
            errors.append(f"line {number}: invalid or missing human_decision")
        if not row.get("reviewer", "").strip():
            errors.append(f"line {number}: reviewer is required")
        expected = system_hash(row, review_fields)
        if row.get("system_row_sha256") != expected:
            errors.append(f"line {number}: system fields were modified")
    if errors:
        raise ValueError("invalid review:\n- " + "\n- ".join(errors))


def apply_reviews(
    policy_rows: list[dict[str, Any]],
    answer_rows: list[dict[str, str]],
    evidence_rows: list[dict[str, str]],
    retrieval_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    validate_review_rows(
        answer_rows,
        allowed_decisions=ANSWER_DECISIONS,
        review_fields={
            "human_decision",
            "corrected_roles",
            "reviewer",
            "comment",
        },
    )
    validate_review_rows(
        evidence_rows,
        allowed_decisions=EVIDENCE_DECISIONS,
        review_fields={"human_decision", "reviewer", "comment"},
    )
    validate_review_rows(
        retrieval_rows,
        allowed_decisions=RETRIEVAL_DECISIONS,
        review_fields={"human_decision", "reviewer", "comment"},
    )
    answers_by_qa: dict[str, list[dict[str, str]]] = {}
    evidence_by_qa: dict[str, list[dict[str, str]]] = {}
    retrieval_by_qa: dict[str, list[dict[str, str]]] = {}
    for row in answer_rows:
        answers_by_qa.setdefault(row["reference_qa_id"], []).append(row)
    for row in evidence_rows:
        evidence_by_qa.setdefault(row["reference_qa_id"], []).append(row)
    for row in retrieval_rows:
        retrieval_by_qa.setdefault(row["reference_qa_id"], []).append(row)

    reviewed = []
    for policy in policy_rows:
        qa_id = policy["reference_qa_id"]
        answer_flow = []
        for row in sorted(
            answers_by_qa.get(qa_id, []),
            key=lambda value: int(value["claim_sequence"]),
        ):
            if row["human_decision"] == "REMOVE":
                continue
            roles = (
                row["corrected_roles"]
                if row["human_decision"] == "CHANGE"
                else row["system_roles"]
            )
            if row["human_decision"] == "CHANGE" and not roles.strip():
                raise ValueError(f"corrected_roles required for {row['review_row_id']}")
            answer_flow.extend(
                value for value in roles.split("|") if value
            )
        retrieval_actions = list(
            dict.fromkeys(
                row["system_action"]
                for row in sorted(
                    retrieval_by_qa.get(qa_id, []),
                    key=lambda value: int(value["sequence"]),
                )
                if row["human_decision"] in {"REQUIRED", "OPTIONAL"}
            )
        )
        evidence_accepted = all(
            row["human_decision"]
            in {"CORRECT", "PARTIAL", "NO_EVIDENCE_APPROPRIATE"}
            for row in evidence_by_qa.get(qa_id, [])
        )
        complete = bool(
            answers_by_qa.get(qa_id)
            and evidence_by_qa.get(qa_id)
            and retrieval_by_qa.get(qa_id)
        )
        production_eligible = (
            policy["source_kind"] == "CUSTOMER_GOLD"
            and complete
            and evidence_accepted
        )
        reviewed.append(
            {
                **policy,
                "pattern_id": pattern_identity(
                    answer_flow, retrieval_actions
                ),
                "answer_flow": answer_flow,
                "retrieval_actions": retrieval_actions,
                "requires_decree": (
                    "FOLLOW_DECREE_DELEGATION" in retrieval_actions
                ),
                "requires_exception_search": (
                    "EXCEPTION_NOTICE" in answer_flow
                ),
                "requires_child_expansion": (
                    "EXPAND_CHILDREN" in retrieval_actions
                ),
                "label_provenance": "HUMAN_REVIEWED",
                "sample_weight": 1.0 if production_eligible else 0.4,
                "production_training_eligible": production_eligible,
            }
        )
    return reviewed
