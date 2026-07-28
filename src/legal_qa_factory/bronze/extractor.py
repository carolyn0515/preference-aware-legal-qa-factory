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


def extract_pdf_blocks(
    pdf_path: Path,
    manifest: dict[str, Any],
    run_id: str,
    extracted_at: datetime,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with fitz.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            text_block_index = 0
            for block in page.get_text("blocks", sort=True):
                if len(block) < 7 or block[6] != 0 or not isinstance(block[4], str):
                    continue
                normalized = normalize_text(block[4])
                if not normalized:
                    continue
                text_hash = sha256_text(normalized)
                page_number = page_index + 1
                records.append(
                    {
                        "bronze_record_id": build_bronze_record_id(
                            manifest["raw_object_id"],
                            page_number,
                            text_block_index,
                            text_hash,
                        ),
                        "raw_object_id": manifest["raw_object_id"],
                        "source_id": manifest["source_id"],
                        "source_type": manifest["source_type"],
                        "content_hash": manifest["file"]["sha256"],
                        "page_number": page_number,
                        "block_index": text_block_index,
                        "parser_block_number": int(block[5]),
                        "bounding_box": {
                            "x0": float(block[0]),
                            "y0": float(block[1]),
                            "x1": float(block[2]),
                            "y1": float(block[3]),
                        },
                        "raw_text": block[4],
                        "normalized_text": normalized,
                        "text_sha256": text_hash,
                        "character_count": len(normalized),
                        "parser_name": "PyMuPDF",
                        "parser_version": fitz.VersionBind,
                        "extraction_run_id": run_id,
                        "extracted_at": extracted_at,
                    }
                )
                text_block_index += 1
    return records
