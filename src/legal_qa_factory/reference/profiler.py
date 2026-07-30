from __future__ import annotations

import re
from collections import Counter
from statistics import mean, median
from typing import Any

import pyarrow as pa

PROFILER_ID = "reference_structural_profile_v1"
ARTICLE_CITATION = re.compile(r"제\d+조(?:의\d+)?")
LAW_TITLE = re.compile(r"「[^」]+」")
LIST_MARKER = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)]|[①-⑳])\s*")

MARKERS = {
    "condition": ("경우", "요건", "해당하면", "때에는"),
    "exception": ("다만", "예외", "제외", "불구하고"),
    "procedure": ("먼저", "다음", "절차", "신청", "통지", "확인"),
    "uncertainty": ("확인해야", "검토해야", "사실관계", "달라질 수"),
    "conclusion": ("해야 합니다", "할 수 있습니다", "아닙니다", "해당합니다"),
}

REFERENCE_FEATURE_SCHEMA = pa.schema(
    [
        pa.field("reference_qa_id", pa.string(), nullable=False),
        pa.field("customer_id", pa.string(), nullable=False),
        pa.field("reference_version", pa.string(), nullable=False),
        pa.field("question_char_count", pa.int32(), nullable=False),
        pa.field("answer_char_count", pa.int32(), nullable=False),
        pa.field("claim_count", pa.int32(), nullable=False),
        pa.field("paragraph_count", pa.int32(), nullable=False),
        pa.field("list_item_count", pa.int32(), nullable=False),
        pa.field("article_citation_count", pa.int32(), nullable=False),
        pa.field("law_title_count", pa.int32(), nullable=False),
        pa.field("has_condition_marker", pa.bool_(), nullable=False),
        pa.field("has_exception_marker", pa.bool_(), nullable=False),
        pa.field("has_procedure_marker", pa.bool_(), nullable=False),
        pa.field("has_uncertainty_marker", pa.bool_(), nullable=False),
        pa.field("has_conclusion_marker", pa.bool_(), nullable=False),
        pa.field("first_claim_has_conclusion", pa.bool_(), nullable=False),
    ],
    metadata={
        b"schema_name": b"reference_qa_observed_feature",
        b"schema_version": b"1.0",
        b"profiler_id": PROFILER_ID.encode(),
    },
)


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def observed_features(
    qa: dict[str, Any], claims: list[dict[str, Any]]
) -> dict[str, Any]:
    answer = qa["answer"]
    first_claim = claims[0]["text"] if claims else ""
    return {
        "reference_qa_id": qa["reference_qa_id"],
        "customer_id": qa["customer_id"],
        "reference_version": qa["reference_version"],
        "question_char_count": len(qa["question"]),
        "answer_char_count": len(answer),
        "claim_count": len(claims),
        "paragraph_count": len(
            [value for value in re.split(r"\n\s*\n", answer) if value.strip()]
        ),
        "list_item_count": len(LIST_MARKER.findall(answer)),
        "article_citation_count": len(ARTICLE_CITATION.findall(answer)),
        "law_title_count": len(LAW_TITLE.findall(answer)),
        "has_condition_marker": _contains_any(answer, MARKERS["condition"]),
        "has_exception_marker": _contains_any(answer, MARKERS["exception"]),
        "has_procedure_marker": _contains_any(answer, MARKERS["procedure"]),
        "has_uncertainty_marker": _contains_any(answer, MARKERS["uncertainty"]),
        "has_conclusion_marker": _contains_any(answer, MARKERS["conclusion"]),
        "first_claim_has_conclusion": _contains_any(
            first_claim, MARKERS["conclusion"]
        ),
    }


def _candidate(
    feature: str, support_count: int, total_count: int
) -> dict[str, Any]:
    rate = support_count / total_count if total_count else 0.0
    if total_count < 10:
        status = "INSUFFICIENT_SAMPLE"
    elif support_count >= 5 and rate >= 0.7:
        status = "CANDIDATE"
    else:
        status = "NOT_ESTABLISHED"
    return {
        "feature": feature,
        "support_count": support_count,
        "total_count": total_count,
        "support_rate": round(rate, 4),
        "status": status,
    }


def aggregate_profile(
    features: list[dict[str, Any]], *, input_sha256: str
) -> dict[str, Any]:
    if not features:
        raise ValueError("cannot profile an empty Reference QA dataset")
    boolean_fields = [
        field.name
        for field in REFERENCE_FEATURE_SCHEMA
        if pa.types.is_boolean(field.type)
    ]
    counts = Counter(
        field for row in features for field in boolean_fields if row[field]
    )
    answer_lengths = [row["answer_char_count"] for row in features]
    claim_counts = [row["claim_count"] for row in features]
    return {
        "schema_version": "1.0",
        "profiler_id": PROFILER_ID,
        "input_sha256": input_sha256,
        "truth_semantics": "PREFERENCE_EVIDENCE_ONLY",
        "customer_id": features[0]["customer_id"],
        "reference_version": features[0]["reference_version"],
        "qa_count": len(features),
        "observed_statistics": {
            "answer_char_count": {
                "minimum": min(answer_lengths),
                "maximum": max(answer_lengths),
                "mean": round(mean(answer_lengths), 2),
                "median": median(answer_lengths),
            },
            "claim_count": {
                "minimum": min(claim_counts),
                "maximum": max(claim_counts),
                "mean": round(mean(claim_counts), 2),
                "median": median(claim_counts),
            },
        },
        "preference_candidates": [
            _candidate(field, counts[field], len(features))
            for field in boolean_fields
        ],
        "interpretation_gate": {
            "minimum_examples": 10,
            "minimum_support_count": 5,
            "minimum_support_rate": 0.7,
            "note": (
                "Candidates describe repeated surface patterns, not legal "
                "correctness or confirmed customer requirements."
            ),
        },
    }
