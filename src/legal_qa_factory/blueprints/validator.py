from __future__ import annotations

from typing import Any


def training_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    customer_gold_count = sum(
        row["source_kind"] == "CUSTOMER_GOLD" for row in rows
    )
    reviewed_count = sum(
        row["label_provenance"] == "HUMAN_REVIEWED" for row in rows
    )
    eligible_count = sum(row["production_training_eligible"] for row in rows)
    blockers = []
    if len(rows) < 10:
        blockers.append("MINIMUM_EXAMPLE_COUNT_NOT_MET")
    if customer_gold_count == 0:
        blockers.append("NO_CUSTOMER_GOLD")
    if reviewed_count < 5:
        blockers.append("INSUFFICIENT_HUMAN_REVIEWED_LABELS")
    if eligible_count < 5:
        blockers.append("INSUFFICIENT_PRODUCTION_ELIGIBLE_ROWS")
    return {
        "status": "BLOCKED" if blockers else "READY",
        "example_count": len(rows),
        "customer_gold_count": customer_gold_count,
        "human_reviewed_count": reviewed_count,
        "production_eligible_count": eligible_count,
        "blockers": blockers,
    }
