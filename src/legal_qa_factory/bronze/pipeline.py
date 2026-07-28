from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from legal_qa_factory.bronze.extractor import extract_pdf_blocks
from legal_qa_factory.bronze.quality import validate_records
from legal_qa_factory.bronze.writer import publish_parquet
from legal_qa_factory.common.hashing import sha256_file
from legal_qa_factory.common.io import load_yaml


def build_bronze(raw_root: Path, bronze_root: Path) -> int:
    run_id = "BRONZE-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    extracted_at = datetime.now(UTC)
    published = 0
    for manifest_path in sorted(raw_root.glob("*/*/manifest.yaml")):
        manifest = load_yaml(manifest_path)
        pdf_path = manifest_path.parent / "original.pdf"
        expected_hash = manifest["file"]["sha256"]
        if sha256_file(pdf_path) != expected_hash:
            raise ValueError(f"Raw object integrity failure: {pdf_path}")
        records = extract_pdf_blocks(pdf_path, manifest, run_id, extracted_at)
        validate_records(records, expected_hash)
        output = bronze_root / manifest["source_id"] / expected_hash
        path = publish_parquet(records, output, manifest, run_id)
        print(f"[PUBLISHED] {manifest['source_id']}: {len(records)} rows -> {path}")
        published += 1
    if not published:
        raise ValueError(f"no Raw manifests found under {raw_root}")
    return published
