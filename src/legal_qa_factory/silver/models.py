from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LegalNode:
    legal_node_id: str
    source_id: str
    source_version_hash: str
    node_type: str
    citation_label: str | None
    text: str
    valid_from: str | None
    valid_to: str | None
    transaction_from: str
    transaction_to: str | None
    bronze_record_ids: tuple[str, ...]
