from pathlib import Path

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any

import fitz
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\u00a0", " ")
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()

def calculate_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()\

def build_bronze_record_id(
    raw_object_id: str,
    page_number: int,
    block_index: int,
    text_sha256: str,
) -> str:

    identity = (
        f"{raw_object_id}|"
        f"{page_number}|"
        f"{block_index}|"
        f"{text_sha256}"
    )

    identity_hash = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()

    return f"BRZ-{identity_hash}"

def load_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = yaml.safe_load(file)
    if not isinstance(manifest, dict):
        raise ValueError(
            f"manifest must be a YAML mapping: {manifest_path}"
        )
    return manifest

def extract_pdf_blocks(
    file_path: Path,
    raw_object_id: str,
    source_id: str,
    source_type: str,
    content_hash: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with fitz.open(file_path) as pdf:
        for page_index, page in enumerate(pdf):
            page_number = page_index + 1
            page_blocks = page.get_text("blocks", sort=True)
            text_block_index = 0
            for block in page_blocks:
                x0, y0, x1, y1, raw_text, _, block_type = block
                if block_type != 0:
                    continue
                normalized_text = normalize_text(raw_text)
                if not normalize_text:
                    continue
                text_sha256 = calculate_text_sha256(
                    normalized_text
                )
                bronze_record_id = build_bronze_record_id(
                    raw_object_id=raw_object_id,
                    page_number=page_number,
                    block_index=text_block_index,
                    text_sha256=text_sha256,
                )
                record = {
                    "bronze_record_id": bronze_record_id,
                    "raw_object_id": raw_object_id,
                    "source_id": source_id,
                    "source_type": source_type,
                    "content_hash": content_hash,
                    "page_number": page_number,
                    "block_index": text_block_index,
                    "bounding_box": {
                        "x0": float(x0),
                        "y0": float(y0),
                        "x1": float(x1),
                        "y1": float(y1),
                    },
                    "raw_text": raw_text,
                    "normalized_text": normalized_text,
                    "text_sha256": text_sha256,
                    "character_count": len(normalized_text),
                    "parser_name": "PyMuPDF",
                    "parser_version": fitz.VersionBind,
                }
                records.append(record)
                text_block_index += 1

    return records

def main() -> int:
    manifest_paths = sorted(
        RAW_DATA_DIR.glob('*/*/manifest.yaml')
    )

    if not manifest_paths:
        print(f"[ERROR] no Raw manifests found: {RAW_DATA_DIR}")
        return 1

    total_records = 0

    for manifest_path in manifest_paths:
        manifest = load_manifest(manifest_path)

        raw_object_id = manifest["raw_object_id"]
        source_id = manifest["source_id"]
        source_type = manifest["source_type"]
        content_hash = manifest["file"]["sha256"]

        original_pdf = manifest_path.parent / "original.pdf"

        if not original_pdf.is_file():
            print(
                f"Raw PDF is missing: {original_pdf}"
            )
            return 1

        records = extract_pdf_blocks(
            file_path=original_pdf,
            raw_object_id=raw_object_id,
            source_id=source_id,
            source_type=source_type,
            content_hash=content_hash,
        )

        total_records += len(records)

        print(f"[SOURCE] {source_id}")
        print(f"[PDF] {original_pdf}")
        print(f"[BLOCKS] {len(records)}")

        for record in records[:3]:
            print()
            print(f"  ID: {record['bronze_record_id']}")
            print(f"  Page: {record['page_number']}")
            print(f"  Block: {record['block_index']}")
            print(f"  Box: {record['bounding_box']}")
            print(
                f"  Text: "
                f"{record['normalized_text'][:100]}"
            )

        print()

    print("[BRONZE EXTRACTION SUMMARY]")
    print(f"Raw objects: {len(manifest_paths)}")
    print(f"Text blocks: {total_records}")
    print("[RESULT] PASSED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())