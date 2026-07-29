from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceLink:
    claim_id: str
    legal_node_id: str
    lineage_kind: str  # OBSERVED or INFERRED
    inference_method: str | None
    confidence: float | None
    rank: int
