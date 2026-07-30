from __future__ import annotations

from typing import Any


def validate_lineage(
    features: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    proposition_ids: set[str],
) -> None:
    feature_ids = {row["reference_claim_id"] for row in features}
    if len(feature_ids) != len(features):
        raise ValueError("duplicate claim features")
    for row in candidates:
        if row["reference_claim_id"] not in feature_ids:
            raise ValueError("candidate references unknown claim")
        if row["evidence_proposition_id"] not in proposition_ids:
            raise ValueError("candidate references unknown proposition")
        if row["lineage_kind"] != "INFERRED":
            raise ValueError("retrieved candidates must use INFERRED lineage")
        if not 0 <= row["final_score"] <= 1:
            raise ValueError("final_score must be in [0, 1]")
