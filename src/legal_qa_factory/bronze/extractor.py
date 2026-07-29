from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz

from legal_qa_factory.bronze.identifiers import build_bronze_record_id
from legal_qa_factory.common.hashing import sha256_text


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFC", text).replace("\u00a0", " ")
    return re.sub(r"\s+", " ", value).strip()


def _box(value: list[float] | tuple[float, ...]) -> dict[str, float]:
    return dict(zip(("x0", "y0", "x1", "y1"), map(float, value), strict=True))


def _is_bold(font_name: str, flags: int) -> bool:
    tokens = ("bold", "black", "heavy", "semibold", "demibold")
    return bool(flags & 16) or any(x in font_name.casefold() for x in tokens)


def _lines(block: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for line_index, line in enumerate(block.get("lines", [])):
        spans = []
        for span_index, span in enumerate(line.get("spans", [])):
            text = str(span.get("text", ""))
            if not text:
                continue
            font = str(span.get("font", "UNKNOWN"))
            flags = int(span.get("flags", 0))
            spans.append(
                {
                    "span_index": span_index,
                    "text": text,
                    "font_name": font,
                    "font_size": float(span.get("size", 0.0)),
                    "font_flags": flags,
                    "is_bold": _is_bold(font, flags),
                    "color": int(span.get("color", 0)),
                    "bounding_box": _box(span["bbox"]),
                }
            )
        if spans:
            result.append(
                {
                    "line_index": line_index,
                    "bounding_box": _box(line["bbox"]),
                    "spans": spans,
                }
            )
    return result


def _trace_runs(page: fitz.Page) -> list[dict[str, Any]]:
    result = []
    for trace_index, trace in enumerate(page.get_texttrace()):
        text = "".join(chr(char[0]) for char in trace["chars"])
        if not text.strip():
            continue
        render_type = int(trace.get("type", 0))
        result.append(
            {
                "trace_index": trace_index,
                "text": text,
                "font_name": str(trace.get("font", "UNKNOWN")),
                "font_size": float(trace.get("size", 0.0)),
                "render_type": render_type,
                "line_width": (
                    float(trace["linewidth"])
                    if trace.get("linewidth") is not None
                    else None
                ),
                "is_simulated_bold": render_type == 1,
                "bounding_box": _box(trace["bbox"]),
            }
        )
    return result


def _overlaps(first: dict[str, float], second: dict[str, float]) -> bool:
    return not (
        first["x1"] < second["x0"]
        or first["x0"] > second["x1"]
        or first["y1"] < second["y0"]
        or first["y0"] > second["y1"]
    )


def extract_pdf_blocks(
    pdf_path: Path,
    manifest: dict[str, Any],
    run_id: str,
    extracted_at: datetime,
) -> list[dict[str, Any]]:
    records = []
    with fitz.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            document = page.get_text("dict", sort=True)
            page_traces = _trace_runs(page)
            block_index = 0
            for parser_number, block in enumerate(document.get("blocks", [])):
                if block.get("type") != 0:
                    continue
                lines = _lines(block)
                raw = "\n".join(
                    "".join(span["text"] for span in line["spans"]) for line in lines
                )
                normalized = normalize_text(raw)
                if not normalized:
                    continue
                text_hash = sha256_text(normalized)
                page_number = page_index + 1
                records.append(
                    {
                        "bronze_record_id": build_bronze_record_id(
                            manifest["raw_object_id"],
                            page_number,
                            block_index,
                            text_hash,
                        ),
                        "raw_object_id": manifest["raw_object_id"],
                        "source_id": manifest["source_id"],
                        "source_type": manifest["source_type"],
                        "content_hash": manifest["file"]["sha256"],
                        "page_number": page_number,
                        "block_index": block_index,
                        "parser_block_number": parser_number,
                        "bounding_box": _box(block["bbox"]),
                        "raw_text": raw,
                        "normalized_text": normalized,
                        "text_sha256": text_hash,
                        "character_count": len(normalized),
                        "lines": lines,
                        "typography_traces": [
                            trace
                            for trace in page_traces
                            if _overlaps(_box(block["bbox"]), trace["bounding_box"])
                        ],
                        "parser_name": "PyMuPDF",
                        "parser_version": fitz.VersionBind,
                        "extraction_run_id": run_id,
                        "extracted_at": extracted_at,
                    }
                )
                block_index += 1
    return records
