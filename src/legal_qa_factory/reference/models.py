from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceQA:
    reference_qa_id: str
    question: str
    answer: str
    customer_id: str
    reference_version: str
    observed_evidence_ids: tuple[str, ...] = ()
