from __future__ import annotations

from statistics import median
from typing import Any


def spans(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [span for line in record["lines"] for span in line["spans"] if span["text"]]


def body_font_size(records: list[dict[str, Any]]) -> float:
    sizes = [
        float(span["font_size"])
        for record in records
        for span in spans(record)
        if not span["is_bold"]
    ]
    return float(median(sizes)) if sizes else 1.0


def heading_evidence(
    record: dict[str, Any], heading_text: str, normal_size: float
) -> tuple[float, tuple[str, ...]]:
    remaining = len(heading_text.replace(" ", ""))
    selected = []
    for span in spans(record):
        if remaining <= 0:
            break
        selected.append(span)
        remaining -= len(span["text"].replace(" ", ""))
    if not selected:
        return 0.0, ()
    evidence = ["LEXICAL_PREFIX"]
    score = 0.60
    simulated_bold = any(
        trace["is_simulated_bold"] and heading_text.startswith(trace["text"].strip())
        for trace in record["typography_traces"]
    )
    if all(span["is_bold"] for span in selected) or simulated_bold:
        score += 0.25
        evidence.append("SIMULATED_BOLD_STROKE" if simulated_bold else "BOLD_HEADING")
    maximum = max(float(span["font_size"]) for span in selected)
    if maximum >= normal_size * 1.05:
        score += 0.10
        evidence.append("LARGER_FONT")
    if record["normalized_text"].startswith(heading_text):
        score += 0.05
        evidence.append("BLOCK_START")
    return min(score, 1.0), tuple(evidence)
