from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BRONZE_ROOT = PROJECT_ROOT / "data" / "bronze"

AUDIT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "artifacts"
    / "article_boundary_audit"
)

SUPPORTED_SOURCE_TYPES = {
    "STATUTE",
    "ENFORCEMENT_DECREE",
}


ARTICLE_PREFIX_PATTERN = re.compile(
    r"^제(?P<article_number>\d+)조"
    r"(?:의(?P<branch_number>\d+))?"
    r"(?P<remainder>.*)$"
)

TITLE_PATTERN = re.compile(
    r"^\((?P<title>[^)]+)\)"
)

DELETED_PATTERN = re.compile(
    r"^\s*삭제(?:\s|<|\[|$)"
)


CSV_FIELDS = [
    "source_id",
    "source_type",
    "content_hash",
    "logical_key",
    "citation_label",
    "article_number",
    "branch_number",
    "title",
    "candidate_type",
    "candidate_status",
    "review_reasons",
    "page_number",
    "block_index",
    "bronze_record_id",
    "normalized_text",
]

def load_bronze_records(
    parquet_path: Path,
) -> list[dict[str, Any]]:
    table = pq.read_table(
        parquet_path,
        columns=[
            "bronze_record_id",
            "source_id",
            "source_type",
            "content_hash",
            "page_number",
            "block_index",
            "normalized_text",
        ],
    )

    records = table.to_pylist()

    records.sort(
        key=lambda record: (
            record["page_number"],
            record["block_index"],
        )
    )

    return records

def build_citation_label(
    article_number: int,
    branch_number: int | None,
) -> str:
    label = f"제{article_number}조"

    if branch_number is not None:
        label += f"의{branch_number}"

    return label


def build_logical_key(
    source_id: str,
    article_number: int,
    branch_number: int | None,
) -> str:
    branch_component = (
        str(branch_number)
        if branch_number is not None
        else "0"
    )

    return (
        f"{source_id}|ARTICLE|"
        f"{article_number}|{branch_component}"
    )

def parse_article_candidate(
    record: dict[str, Any],
) -> dict[str, Any] | None:
    text = record["normalized_text"].strip()

    match = ARTICLE_PREFIX_PATTERN.match(text)

    if match is None:
        return None

    article_number = int(
        match.group("article_number")
    )

    branch_value = match.group("branch_number")
    branch_number = (
        int(branch_value)
        if branch_value is not None
        else None
    )

    remainder = match.group("remainder").lstrip()

    title_match = TITLE_PATTERN.match(remainder)
    deleted_match = DELETED_PATTERN.match(remainder)

    title = (
        title_match.group("title").strip()
        if title_match is not None
        else None
    )

    if deleted_match is not None:
        candidate_status = "DELETED"
    elif title is not None:
        candidate_status = "REGULAR"
    else:
        candidate_status = "SUSPICIOUS"

    candidate_type = (
        "ARTICLE_BRANCH"
        if branch_number is not None
        else "ARTICLE"
    )

    return {
        "source_id": record["source_id"],
        "source_type": record["source_type"],
        "content_hash": record["content_hash"],
        "logical_key": build_logical_key(
            source_id=record["source_id"],
            article_number=article_number,
            branch_number=branch_number,
        ),
        "citation_label": build_citation_label(
            article_number=article_number,
            branch_number=branch_number,
        ),
        "article_number": article_number,
        "branch_number": branch_number,
        "title": title,
        "candidate_type": candidate_type,
        "candidate_status": candidate_status,
        "review_reasons": [],
        "page_number": record["page_number"],
        "block_index": record["block_index"],
        "bronze_record_id": record["bronze_record_id"],
        "normalized_text": text,
    }

def extract_article_candidates(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for record in records:
        candidate = parse_article_candidate(record)

        if candidate is not None:
            candidates.append(candidate)

    return candidates

def add_review_reason(
    candidate: dict[str, Any],
    reason: str,
) -> None:
    reasons = candidate["review_reasons"]

    if reason not in reasons:
        reasons.append(reason)

def mark_duplicate_keys(
    candidates: list[dict[str, Any]],
) -> list[str]:
    candidates_by_key: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for candidate in candidates:
        candidates_by_key[
            candidate["logical_key"]
        ].append(candidate)

    duplicate_keys: list[str] = []

    for logical_key, grouped_candidates in (
        candidates_by_key.items()
    ):
        if len(grouped_candidates) <= 1:
            continue

        duplicate_keys.append(logical_key)

        for candidate in grouped_candidates:
            add_review_reason(
                candidate,
                "DUPLICATE_LOGICAL_KEY",
            )

    return sorted(duplicate_keys)

def article_sort_key(
    candidate: dict[str, Any],
) -> tuple[int, int]:
    return (
        candidate["article_number"],
        candidate["branch_number"] or 0,
    )


def mark_non_monotonic_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, str]]:
    transitions: list[dict[str, str]] = []

    previous: dict[str, Any] | None = None

    for current in candidates:
        if previous is None:
            previous = current
            continue

        previous_key = article_sort_key(previous)
        current_key = article_sort_key(current)

        if current_key <= previous_key:
            add_review_reason(
                current,
                "NON_MONOTONIC_ARTICLE_ORDER",
            )

            transitions.append(
                {
                    "previous": previous[
                        "citation_label"
                    ],
                    "current": current[
                        "citation_label"
                    ],
                }
            )

        previous = current

    return transitions

def mark_suspicious_candidates(
    candidates: list[dict[str, Any]],
) -> None:
    for candidate in candidates:
        if candidate["candidate_status"] != "SUSPICIOUS":
            continue

        add_review_reason(
            candidate,
            "MISSING_ARTICLE_TITLE",
        )

def find_main_article_gaps(
    candidates: list[dict[str, Any]],
) -> list[int]:
    main_article_numbers = sorted(
        {
            candidate["article_number"]
            for candidate in candidates
            if candidate["branch_number"] is None
        }
    )

    if not main_article_numbers:
        return []

    first_number = main_article_numbers[0]
    last_number = main_article_numbers[-1]

    observed = set(main_article_numbers)

    return [
        number
        for number in range(
            first_number,
            last_number + 1,
        )
        if number not in observed
    ]

def find_branch_gaps(
    candidates: list[dict[str, Any]],
) -> dict[int, list[int]]:
    branches_by_article: dict[
        int,
        set[int],
    ] = defaultdict(set)

    for candidate in candidates:
        branch_number = candidate["branch_number"]

        if branch_number is None:
            continue

        branches_by_article[
            candidate["article_number"]
        ].add(branch_number)

    gaps: dict[int, list[int]] = {}

    for article_number, branch_numbers in (
        branches_by_article.items()
    ):
        minimum = min(branch_numbers)
        maximum = max(branch_numbers)

        missing = [
            number
            for number in range(minimum, maximum + 1)
            if number not in branch_numbers
        ]

        if missing:
            gaps[article_number] = missing

    return gaps

def serialize_candidate(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    serialized = dict(candidate)

    serialized["branch_number"] = (
        candidate["branch_number"]
        if candidate["branch_number"] is not None
        else ""
    )

    serialized["title"] = (
        candidate["title"]
        if candidate["title"] is not None
        else ""
    )

    serialized["review_reasons"] = "|".join(
        candidate["review_reasons"]
    )

    return serialized


def write_candidates_csv(
    candidates: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()

        for candidate in candidates:
            writer.writerow(
                serialize_candidate(candidate)
            )

    temporary_path.replace(output_path)

def write_summary_yaml(
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            summary,
            file,
            allow_unicode=True,
            sort_keys=False,
        )

    temporary_path.replace(output_path)

def audit_source(
    parquet_path: Path,
) -> dict[str, Any]:
    records = load_bronze_records(parquet_path)

    if not records:
        raise ValueError(
            f"empty Bronze dataset: {parquet_path}"
        )

    source_id = records[0]["source_id"]
    source_type = records[0]["source_type"]
    content_hash = records[0]["content_hash"]

    if source_type not in SUPPORTED_SOURCE_TYPES:
        return {
            "source_id": source_id,
            "status": "SKIPPED",
            "reason": (
                f"unsupported source type: "
                f"{source_type}"
            ),
        }

    candidates = extract_article_candidates(records)

    duplicate_keys = mark_duplicate_keys(candidates)

    non_monotonic_transitions = (
        mark_non_monotonic_candidates(candidates)
    )

    mark_suspicious_candidates(candidates)

    main_article_gaps = find_main_article_gaps(
        candidates
    )

    branch_gaps = find_branch_gaps(candidates)

    review_queue = [
        candidate
        for candidate in candidates
        if candidate["review_reasons"]
    ]

    candidate_status_counts = Counter(
        candidate["candidate_status"]
        for candidate in candidates
    )

    candidate_type_counts = Counter(
        candidate["candidate_type"]
        for candidate in candidates
    )

    output_dir = (
        AUDIT_ROOT
        / source_id
        / content_hash
    )

    write_candidates_csv(
        candidates,
        output_dir / "article_candidates.csv",
    )

    write_candidates_csv(
        review_queue,
        output_dir / "review_queue.csv",
    )

    summary = {
        "schema_version": "1.0",
        "audit_name": "article_boundary_audit",
        "source_id": source_id,
        "source_type": source_type,
        "source_content_hash": content_hash,
        "bronze_path": str(parquet_path),
        "bronze_record_count": len(records),
        "article_candidate_count": len(candidates),
        "distinct_logical_key_count": len(
            {
                candidate["logical_key"]
                for candidate in candidates
            }
        ),
        "candidate_type_counts": dict(
            candidate_type_counts
        ),
        "candidate_status_counts": dict(
            candidate_status_counts
        ),
        "review_candidate_count": len(
            review_queue
        ),
        "duplicate_logical_keys": duplicate_keys,
        "non_monotonic_transitions": (
            non_monotonic_transitions
        ),
        "missing_main_article_numbers": (
            main_article_gaps
        ),
        "missing_branch_numbers": branch_gaps,
        "outputs": {
            "all_candidates": str(
                output_dir
                / "article_candidates.csv"
            ),
            "review_queue": str(
                output_dir
                / "review_queue.csv"
            ),
        },
        "interpretation": {
            "candidate_is_not_ground_truth": True,
            "gap_is_not_automatically_an_error": True,
            "manual_review_required_for": [
                "duplicate logical keys",
                "non-monotonic order",
                "missing titles",
                "main article gaps",
                "branch number gaps",
            ],
        },
    }

    write_summary_yaml(
        summary,
        output_dir / "summary.yaml",
    )

    return {
        "source_id": source_id,
        "status": "AUDITED",
        "article_candidate_count": len(candidates),
        "review_candidate_count": len(review_queue),
        "output_dir": str(output_dir),
    }

def main() -> int:
    parquet_paths = sorted(
        BRONZE_ROOT.glob("*/*/records.parquet")
    )

    if not parquet_paths:
        print(
            "[ERROR] no Bronze Parquet files found"
        )
        return 1

    audited_count = 0

    print("[ARTICLE BOUNDARY AUDIT]")
    print()

    for parquet_path in parquet_paths:
        result = audit_source(parquet_path)

        print(f"[SOURCE] {result['source_id']}")
        print(f"[STATUS] {result['status']}")

        if result["status"] == "AUDITED":
            print(
                "[ARTICLE CANDIDATES] "
                f"{result['article_candidate_count']}"
            )
            print(
                "[REVIEW CANDIDATES] "
                f"{result['review_candidate_count']}"
            )
            print(
                f"[OUTPUT] {result['output_dir']}"
            )

            audited_count += 1
        else:
            print(
                f"[REASON] {result['reason']}"
            )

        print()

    if audited_count == 0:
        print(
            "[RESULT] FAILED: "
            "no statute source was audited"
        )
        return 1

    print(
        f"[RESULT] PASSED: "
        f"{audited_count} sources audited"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
