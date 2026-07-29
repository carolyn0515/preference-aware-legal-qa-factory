from __future__ import annotations

import re
from collections import Counter
from typing import Any

PAGE_NUMBER = re.compile(r"\d+")
KNOWN_FOOTER = re.compile(r"^법제처\s+\d+\s+국가법령정보센터$")


def _signature(text: str) -> str:
    return PAGE_NUMBER.sub("<PAGE>", " ".join(text.split()))


def remove_repeated_boilerplate(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    page_count = len({record["page_number"] for record in records})
    frequency = Counter(_signature(record["normalized_text"]) for record in records)
    page_max_y = {}
    for record in records:
        page = record["page_number"]
        page_max_y[page] = max(
            page_max_y.get(page, 0.0),
            float(record["bounding_box"]["y1"]),
        )

    kept = []
    excluded = []
    for record in records:
        text = record["normalized_text"]
        signature = _signature(text)
        page = record["page_number"]
        relative_y = float(record["bounding_box"]["y0"]) / page_max_y[page]
        repeated = frequency[signature] >= max(2, round(page_count * 0.5))
        bottom_position = relative_y >= 0.90
        top_position = relative_y <= 0.10
        reasons = []
        if KNOWN_FOOTER.fullmatch(text):
            reasons.append("KNOWN_LEGAL_SOURCE_FOOTER")
        if repeated and bottom_position:
            reasons.append("REPEATED_BOTTOM_SIGNATURE")
        if repeated and top_position:
            reasons.append("REPEATED_TOP_SIGNATURE")
        if reasons:
            excluded.append(
                {
                    "bronze_record_id": record["bronze_record_id"],
                    "page_number": page,
                    "block_index": record["block_index"],
                    "text": text,
                    "signature": signature,
                    "relative_y": relative_y,
                    "reasons": reasons,
                }
            )
        else:
            kept.append(record)
    return kept, excluded
