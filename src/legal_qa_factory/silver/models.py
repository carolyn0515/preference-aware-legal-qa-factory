from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Candidate:
    record: dict[str, Any]
    node_type: str | None
    marker: str | None
    citation: str | None
    title: str | None
    body: str
    confidence: float
    evidence: tuple[str, ...]
    region: str


@dataclass
class LegalNode:
    legal_node_id: str
    source_id: str
    source_type: str
    source_version_hash: str
    node_type: str
    parent_node_id: str | None
    article_node_id: str
    sequence: int
    citation_label: str
    title: str | None
    marker: str
    text: str
    region: str
    page_from: int
    page_to: int
    bronze_record_ids: list[str]
    boundary_confidence: float
    boundary_evidence: list[str]
    parser_id: str
    transaction_from: object
    transaction_to: object | None = None

    def as_dict(self) -> dict[str, Any]:
        return vars(self)
