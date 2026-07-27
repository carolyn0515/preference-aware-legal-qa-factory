import hashlib
import json
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fitz
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
BRONZE_DATA_DIR = PROJECT_ROOT / "data" / "bronze"

def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\u00a0", " ")
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()

def calculate_text_sha256(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

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

def load_manifest(
    manifest_path: Path,
) -> dict[str, Any]:
    with manifest_path.open(
        "r",
        encoding="utf-8"
    ) as file:
        manifest = yaml.safe_load(file)

    if not isinstance(manifest, dict):
        raise ValueError(
            "manifest must be a YAML mapping: "
            f"{manifest_path}"
        )

    required_fields = {
        "raw_object_id",
        "source_id",
        "source_type",
        "file",
    }

    for field in required_fields:
        if field not in manifest:
            raise ValueError(
                f"missing manifest field '{field}': "
                f"{manifest_path}"
            )

    file_metadata = manifest["file"]

    if not isinstance(file_metadata, dict):
        raise ValueError(
            "manifest field 'file' must be a mapping: "
            f"{manifest_path}"
        )

    if "sha256" not in file_metadata:
        raise ValueError(
            "missing manifest field 'file.sha256': "
            f"{manifest_path}"
        )
    
    return manifest

def extract_pdf_blocks(
    file_path: Path,
    raw_object_id: str,
    source_id: str,
    source_type: str,
    content_hash: str,
    extraction_run_id: str,
    extracted_at: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    try:
        with fitz.open(file_path) as pdf:
            for page_index, page in enumerate(pdf):
                page_number = page_index + 1
                page_blocks = page.get_text(
                    "blocks",
                    sort=True,
                )
                text_block_index = 0
                for block in page_blocks:
                    if len(block) < 7:
                        continue
                    (
                        x0,
                        y0,
                        x1,
                        y1,
                        raw_text,
                        parser_block_number,
                        block_type,
                    ) = block[:7]

                    if block_type != 0:
                        continue

                    if not isinstance(raw_text, str):
                        continue

                    normalized_text = normalize_text(
                        raw_text
                    )

                    if not normalized_text:
                        continue

                    text_sha256 = (
                        calculate_text_sha256(
                            normalized_text
                        )
                    )

                    bronze_record_id = (
                        build_bronze_record_id(
                            raw_object_id=raw_object_id,
                            page_number=page_number,
                            block_index=text_block_index,
                            text_sha256=text_sha256,
                        )
                    )

                    record = {
                        "bronze_record_id": (
                            bronze_record_id
                        ),
                        "raw_object_id": raw_object_id,
                        "source_id": source_id,
                        "source_type": source_type,
                        "content_hash": content_hash,
                        "page_number": page_number,
                        "block_index": (
                            text_block_index
                        ),
                        "parser_block_number": int(
                            parser_block_number
                        ),
                        "bounding_box": {
                            "x0": float(x0),
                            "y0": float(y0),
                            "x1": float(x1),
                            "y1": float(y1)
                        },
                        "raw_text": raw_text,
                        "normalized_text": (
                            normalized_text
                        ),
                        "text_sha256": text_sha256,
                        "character_count": len(
                            normalized_text
                        ),
                        "parser_name": "PyMuPDF",
                        "parser_version": (
                            fitz.VersionBind
                        ),
                        "extraction_run_id": (
                            extraction_run_id
                        ),
                        "extracted_at": extracted_at,
                    }

                    records.append(record)
                    text_block_index += 1

    except (
        fitz.FileDataError,
        RuntimeError,
        ValueError,
    ) as error:
        raise RuntimeError(
            f"failed to parse PDF: {file_path}"
        ) from error

    return records

def write_jsonl(
        records: list[dict[str, Any]],
        output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )
    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            line = json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":")
            )

            file.write(line)
            file.write("\n")

    temporary_path.replace(output_path)

def build_output_path(
    source_id: str,
    content_hash: str,
) -> Path:
    return (
        BRONZE_DATA_DIR
        / source_id
        / content_hash
        / "records.jsonl"
    )

def print_record_samples(
    records: list[dict[str, Any]],
    sample_size: int = 3,
) -> None:
    for record in records[:sample_size]:
        print()
        print(
            "  ID: "
            f"{record['bronze_record_id']}"
        )
        print(
            "  Page: "
            f"{record['page_number']}"
        )
        print(
            "  Block: "
            f"{record['block_index']}"
        )
        print(
            "  Box: "
            f"{record['bounding_box']}"
        )
        print(
            "  Text: "
            f"{record['normalized_text'][:100]}"
        )

def process_manifest(
    manifest_path: Path,
    extraction_run_id: str,
    extracted_at: str,
) -> int:
    manifest = load_manifest(manifest_path)
    raw_object_id = str(
        manifest["raw_object_id"]
    )
    source_id = str(
        manifest["source_id"]
    )
    source_type = str(
        manifest["source_type"]
    )
    content_hash = str(
        manifest["file"]["sha256"]
    )

    original_pdf = (
        manifest_path.parent
        / "original.pdf"
    )

    if not original_pdf.is_file():
        raise FileNotFoundError(
            f"Raw PDF is missing: {original_pdf}"
        )

    records = extract_pdf_blocks(
        file_path=original_pdf,
        raw_object_id=raw_object_id,
        source_id=source_id,
        source_type=source_type,
        content_hash=content_hash,
        extraction_run_id=extraction_run_id,
        extracted_at=extracted_at,
    )

    output_path = build_output_path(
        source_id=source_id,
        content_hash=content_hash,
    )

    write_jsonl(
        records=records,
        output_path=output_path,
    )

    print(f"[SOURCE] {source_id}")
    print(f"[RAW OBJECT] {raw_object_id}")
    print(f"[PDF] {original_pdf}")
    print(f"[BRONZE] {output_path}")
    print(f"[BLOCKS] {len(records)}")

    print_record_samples(records)
    print()

    return len(records)

def main() -> int:
    extraction_started_at = datetime.now(
        UTC
    )
    extraction_run_id = (
        "BRONZE-"
        + extraction_started_at.strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
    )
    extracted_at = extraction_started_at.isoformat()

    manifest_paths = sorted(
        RAW_DATA_DIR.glob(
            "*/*/manifest.yaml"
        )
    )

    if not manifest_paths:
        print(
            "[ERROR] no Raw manifests found: "
            f"{RAW_DATA_DIR}"
        )
        return 1

    total_records = 0
    processed_objects = 0
    failed_objects = 0

    print("[LEGAL QA FACTORY — BRONZE EXTRACTION]")
    print(f"[RUN] {extraction_run_id}")
    print()

    for manifest_path in manifest_paths:
        try:
            record_count = process_manifest(
                manifest_path=manifest_path,
                extraction_run_id=extraction_run_id,
                extracted_at=extracted_at,
            )
            total_records += record_count
            processed_objects += 1

        except (
            FileNotFoundError,
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
            OSError,
        ) as error:
            failed_objects += 1

            print(
                "[ERROR] failed to process "
                f"{manifest_path}"
            )
            print(f"[CAUSE] {error}")
            print()

    print("[BRONZE EXTRACTION SUMMARY]")
    print(
        f"Raw objects discovered: "
        f"{len(manifest_paths)}"
    )
    print(
        f"Raw objects processed: "
        f"{processed_objects}"
    )
    print(
        f"Raw objects failed: "
        f"{failed_objects}"
    )
    print(
        f"Text blocks created: "
        f"{total_records}"
    )

    if failed_objects > 0:
        print("[RESULT] FAILED")
        return 1

    print("[RESULT] PASSED")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
