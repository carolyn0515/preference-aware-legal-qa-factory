from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from legal_qa_factory.common.io import atomic_yaml_dump

REVIEW_COLUMNS = (
    "review_status",
    "reviewer",
    "reviewed_at",
    "correct_labels",
    "correct_subject",
    "correct_action",
    "correct_object",
    "evidence_valid",
    "review_comment",
)


def _stable_order(value: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _length_bucket(text: str) -> str:
    length = len(text)
    if length < 80:
        return "SHORT"
    if length < 200:
        return "MEDIUM"
    return "LONG"


def load_corpus(silver_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for proposition_path in sorted(silver_dir.glob("*/*/propositions.parquet")):
        nodes = pq.read_table(
            proposition_path.parent / "legal_nodes.parquet"
        ).to_pylist()
        nodes_by_id = {row["legal_node_id"]: row for row in nodes}
        function_path = proposition_path.parent / "legal_functions.parquet"
        functions = (
            {
                row["proposition_id"]: row
                for row in pq.read_table(function_path).to_pylist()
            }
            if function_path.exists()
            else {}
        )
        for proposition in pq.read_table(proposition_path).to_pylist():
            node = nodes_by_id[proposition["legal_node_id"]]
            function = functions.get(proposition["proposition_id"], {})
            labels = function.get("labels") or []
            intrinsic = function.get("intrinsic_labels")
            inherited = function.get("inherited_labels")
            rows.append(
                {
                    **proposition,
                    "source_type": node["source_type"],
                    "node_type": node["node_type"],
                    "citation_label": node["citation_label"],
                    "article_title": nodes_by_id[
                        node["article_node_id"]
                    ].get("title"),
                    "length_bucket": _length_bucket(proposition["text"]),
                    "classified": bool(function),
                    "labels": labels,
                    "intrinsic_labels": intrinsic if intrinsic is not None else labels,
                    "inherited_labels": inherited or [],
                    "semantic_unit_type": function.get(
                        "semantic_unit_type", "LEGACY_UNKNOWN"
                    ),
                    "subject": function.get("subject"),
                    "action": function.get("action"),
                    "object": function.get("object"),
                    "modality": function.get("modality"),
                    "evidence_phrases": function.get("evidence_phrases") or [],
                    "confidence": function.get("confidence"),
                    "classification_method": function.get(
                        "classification_method"
                    ),
                    "provider": function.get("provider"),
                    "model": function.get("model"),
                    "prompt_id": function.get("prompt_id"),
                }
            )
    return rows


def stratified_sample(
    rows: list[dict[str, Any]],
    sample_size: int,
    *,
    seed: str = "legal-function-review-v1",
) -> list[dict[str, Any]]:
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")
    strata: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["source_id"],
            row["node_type"],
            row["length_bucket"],
        )
        strata[key].append(row)
    for values in strata.values():
        values.sort(key=lambda row: _stable_order(row["proposition_id"], seed))

    selected: list[dict[str, Any]] = []
    keys = sorted(strata)
    while len(selected) < min(sample_size, len(rows)):
        progressed = False
        for key in keys:
            if strata[key] and len(selected) < sample_size:
                selected.append(strata[key].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def suspicious_reasons(row: dict[str, Any]) -> list[str]:
    if not row["classified"]:
        return []
    intrinsic = set(row["intrinsic_labels"])
    reasons = []
    if "UNCLASSIFIED" in intrinsic and len(intrinsic) > 1:
        reasons.append("UNCLASSIFIED_WITH_OTHER_LABEL")
    if "OBLIGATION" in intrinsic and not row["action"]:
        reasons.append("OBLIGATION_WITHOUT_ACTION")
    if row["semantic_unit_type"] == "LIST_FRAGMENT" and intrinsic & {
        "DEFINITION",
        "SCOPE",
    }:
        reasons.append("LIST_FRAGMENT_HAS_CONTEXTUAL_LABEL")
    if not row["evidence_phrases"]:
        reasons.append("MISSING_EVIDENCE")
    if row["confidence"] is not None and row["confidence"] < 0.7:
        reasons.append("LOW_CONFIDENCE")
    return reasons


def _csv_value(value: Any) -> Any:
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def build_audit(
    *,
    silver_dir: Path,
    output_dir: Path,
    sample_size: int,
) -> dict[str, Any]:
    rows = load_corpus(silver_dir)
    classified = [row for row in rows if row["classified"]]
    prompt_ids = sorted({row["prompt_id"] for row in classified if row["prompt_id"]})
    schema_generations = sorted(
        {
            "CURRENT"
            if row["semantic_unit_type"] != "LEGACY_UNKNOWN"
            else "LEGACY"
            for row in classified
        }
    )
    version_consistent = len(prompt_ids) <= 1 and len(schema_generations) <= 1

    effective = Counter(label for row in classified for label in row["labels"])
    intrinsic = Counter(
        label for row in classified for label in row["intrinsic_labels"]
    )
    inherited = Counter(
        label for row in classified for label in row["inherited_labels"]
    )
    suspicious = []
    for row in classified:
        reasons = suspicious_reasons(row)
        if reasons:
            suspicious.append({**row, "suspicious_reasons": reasons})

    classification_sample = stratified_sample(rows, sample_size)
    review_sample = stratified_sample(
        classified,
        min(sample_size, len(classified)),
        seed="legal-function-human-review-v1",
    ) if classified else []
    review_rows = [
        {
            **row,
            **{field: "" for field in REVIEW_COLUMNS},
            "predicted_labels": row["labels"],
        }
        for row in review_sample
    ]
    common_fields = [
        "source_id",
        "source_version_hash",
        "proposition_id",
        "legal_node_id",
        "citation_label",
        "article_title",
        "node_type",
        "semantic_unit_type",
        "text",
        "labels",
        "intrinsic_labels",
        "inherited_labels",
        "subject",
        "action",
        "object",
        "modality",
        "evidence_phrases",
        "confidence",
        "prompt_id",
    ]
    write_csv(output_dir / "classification_preview.csv", classified, common_fields)
    write_csv(
        output_dir / "unclassified_cases.csv",
        [row for row in classified if row["labels"] == ["UNCLASSIFIED"]],
        common_fields,
    )
    write_csv(
        output_dir / "multi_label_cases.csv",
        [row for row in classified if len(row["labels"]) > 1],
        common_fields,
    )
    write_csv(
        output_dir / "inheritance_cases.csv",
        [row for row in classified if row["inherited_labels"]],
        common_fields,
    )
    write_csv(
        output_dir / "suspicious_cases.csv",
        suspicious,
        [*common_fields, "suspicious_reasons"],
    )
    write_csv(
        output_dir / "classification_sample.csv",
        classification_sample,
        [
            "source_id",
            "source_version_hash",
            "proposition_id",
            "legal_node_id",
            "citation_label",
            "article_title",
            "node_type",
            "length_bucket",
            "text",
        ],
    )
    write_csv(
        output_dir / "human_review_template.csv",
        review_rows,
        [
            "source_id",
            "proposition_id",
            "citation_label",
            "article_title",
            "node_type",
            "semantic_unit_type",
            "length_bucket",
            "text",
            "predicted_labels",
            "subject",
            "action",
            "object",
            "evidence_phrases",
            "confidence",
            *REVIEW_COLUMNS,
        ],
    )
    report = {
        "audit_version": "1.0",
        "status": "BLOCKED_MIXED_VERSION" if not version_consistent else "READY",
        "version_consistent": version_consistent,
        "prompt_ids": prompt_ids,
        "schema_generations": schema_generations,
        "corpus_count": len(rows),
        "classified_count": len(classified),
        "unclassified_corpus_count": len(rows) - len(classified),
        "classification_sample_count": len(classification_sample),
        "review_sample_count": len(review_sample),
        "suspicious_count": len(suspicious),
        "label_distribution": {
            "effective": dict(sorted(effective.items())),
            "intrinsic": dict(sorted(intrinsic.items())),
            "inherited": dict(sorted(inherited.items())),
        },
        "evaluation_allowed": version_consistent,
        "blocking_reason": (
            None
            if version_consistent
            else "Prompt/schema versions are mixed; do not aggregate quality metrics."
        ),
    }
    atomic_yaml_dump(report, output_dir / "audit_report.yaml")
    return report
