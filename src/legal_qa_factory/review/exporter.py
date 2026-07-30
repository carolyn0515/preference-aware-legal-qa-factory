from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from legal_qa_factory.common.hashing import sha256_text

ANSWER_FIELDS = [
    "review_row_id",
    "reference_qa_id",
    "reference_claim_id",
    "claim_sequence",
    "claim_text",
    "system_roles",
    "system_row_sha256",
    "human_decision",
    "corrected_roles",
    "reviewer",
    "comment",
]
EVIDENCE_FIELDS = [
    "review_row_id",
    "reference_qa_id",
    "reference_claim_id",
    "claim_text",
    "evidence_proposition_id",
    "source_id",
    "article_citation_label",
    "citation_label",
    "evidence_text",
    "relation",
    "score",
    "system_decision",
    "system_row_sha256",
    "human_decision",
    "reviewer",
    "comment",
]
RETRIEVAL_FIELDS = [
    "review_row_id",
    "reference_qa_id",
    "reference_claim_id",
    "sequence",
    "system_action",
    "source_id",
    "article_citation_label",
    "citation_label",
    "relation",
    "system_row_sha256",
    "human_decision",
    "reviewer",
    "comment",
]


def system_hash(row: dict[str, Any], review_fields: set[str]) -> str:
    value = {
        key: "" if row[key] is None else str(row[key])
        for key in sorted(row)
        if key not in review_fields and key != "system_row_sha256"
    }
    return sha256_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _finalize(
    row: dict[str, Any], prefix: str, review_fields: set[str]
) -> dict[str, Any]:
    identity = json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    row["review_row_id"] = f"{prefix}-{sha256_text(identity)[:24]}"
    row["system_row_sha256"] = system_hash(row, review_fields)
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_review_batch(dataset_dir: Path, output_dir: Path) -> dict[str, Any]:
    claims = pq.read_table(dataset_dir / "reference_claims.parquet").to_pylist()
    features = pq.read_table(
        dataset_dir / "lineage" / "claim_features.parquet"
    ).to_pylist()
    candidates = pq.read_table(
        dataset_dir / "lineage" / "claim_evidence_candidates.parquet"
    ).to_pylist()
    expansions = pq.read_table(
        dataset_dir / "lineage" / "claim_evidence_expansions.parquet"
    ).to_pylist()
    selected_by_claim: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        if row["selected"]:
            selected_by_claim.setdefault(row["reference_claim_id"], []).append(row)

    answer_review_fields = {"human_decision", "corrected_roles", "reviewer", "comment"}
    answer_rows = []
    for feature in features:
        row = {
            "reference_qa_id": feature["reference_qa_id"],
            "reference_claim_id": feature["reference_claim_id"],
            "claim_sequence": feature["claim_sequence"],
            "claim_text": feature["text"],
            "system_roles": "|".join(feature["answer_roles"]),
            "human_decision": "",
            "corrected_roles": "",
            "reviewer": "",
            "comment": "",
        }
        answer_rows.append(
            _finalize(row, "ARV", answer_review_fields)
        )

    evidence_review_fields = {"human_decision", "reviewer", "comment"}
    evidence_rows = []
    for claim in claims:
        selected = selected_by_claim.get(claim["reference_claim_id"], [])
        if not selected:
            selected = [
                {
                    "evidence_proposition_id": "",
                    "source_id": "",
                    "article_citation_label": "",
                    "citation_label": "",
                    "evidence_text": "",
                    "retrieval_relation": "NO_DIRECT_LEGAL_EVIDENCE",
                    "final_score": 0.0,
                    "selection_status": "NO_DIRECT_LEGAL_EVIDENCE",
                }
            ]
        for candidate in selected:
            row = {
                "reference_qa_id": claim["reference_qa_id"],
                "reference_claim_id": claim["reference_claim_id"],
                "claim_text": claim["text"],
                "evidence_proposition_id": candidate[
                    "evidence_proposition_id"
                ],
                "source_id": candidate["source_id"],
                "article_citation_label": candidate[
                    "article_citation_label"
                ],
                "citation_label": candidate["citation_label"],
                "evidence_text": candidate["evidence_text"],
                "relation": candidate["retrieval_relation"],
                "score": round(float(candidate["final_score"]), 6),
                "system_decision": candidate["selection_status"],
                "human_decision": "",
                "reviewer": "",
                "comment": "",
            }
            evidence_rows.append(
                _finalize(row, "ERV", evidence_review_fields)
            )

    relation_actions = {
        "DIRECT_LEXICAL": "SEARCH_ANCHOR",
        "CHILD_ENUMERATION": "EXPAND_CHILDREN",
        "PARENT_CONTEXT": "EXPAND_PARENT",
        "SAME_ARTICLE_ROLE": "SEARCH_ROLE_SIBLINGS",
        "REFERENCED_ARTICLE": "FOLLOW_ARTICLE_REFERENCE",
        "IMPLEMENTING_DECREE": "FOLLOW_DECREE_DELEGATION",
    }
    retrieval_review_fields = {"human_decision", "reviewer", "comment"}
    retrieval_rows = []
    sequence_by_qa: dict[str, int] = {}
    selected_candidates = [row for row in candidates if row["selected"]]
    traversal_rows = [
        {
            "reference_qa_id": row["reference_qa_id"],
            "reference_claim_id": row["reference_claim_id"],
            "source_id": row["source_id"],
            "article_citation_label": row["article_citation_label"],
            "citation_label": row["citation_label"],
            "relation": row["retrieval_relation"],
        }
        for row in selected_candidates
    ] + [
        {
            "reference_qa_id": row["reference_qa_id"],
            "reference_claim_id": row["reference_claim_id"],
            "source_id": row["source_id"],
            "article_citation_label": row["article_citation_label"],
            "citation_label": row["citation_label"],
            "relation": row["expansion_relation"],
        }
        for row in expansions
    ]
    for source in traversal_rows:
        qa_id = source["reference_qa_id"]
        sequence_by_qa[qa_id] = sequence_by_qa.get(qa_id, 0) + 1
        row = {
            **source,
            "sequence": sequence_by_qa[qa_id],
            "system_action": relation_actions[source["relation"]],
            "human_decision": "",
            "reviewer": "",
            "comment": "",
        }
        retrieval_rows.append(
            _finalize(row, "RRV", retrieval_review_fields)
        )

    _write_csv(output_dir / "answer_flow_review.csv", answer_rows, ANSWER_FIELDS)
    _write_csv(
        output_dir / "claim_evidence_review.csv",
        evidence_rows,
        EVIDENCE_FIELDS,
    )
    _write_csv(
        output_dir / "retrieval_flow_review.csv",
        retrieval_rows,
        RETRIEVAL_FIELDS,
    )
    return {
        "answer_review_count": len(answer_rows),
        "evidence_review_count": len(evidence_rows),
        "retrieval_review_count": len(retrieval_rows),
    }
