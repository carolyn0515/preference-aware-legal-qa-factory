from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fitz
import yaml

from legal_qa_factory.common.hashing import sha256_file
from legal_qa_factory.common.io import load_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "sources"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
QUARANTINE_DIR = PROJECT_ROOT / "data" / "quarantine"

def resolve_input_path(config: dict[str, Any]) -> Path:
    configured_path = Path(config["file"]["input_path"])
    if configured_path.is_absolute():
        return configured_path
    return PROJECT_ROOT / configured_path

def inspect_pdf(file_path: Path) -> int:
    with fitz.open(file_path) as pdf:
        if pdf.page_count < 1:
            raise ValueError("PDF has no pages")
        return pdf.page_count

def quarantine_file(
    file_path: Path,
    source_id: str,
    content_hash: str,
    error: Exception,
) -> Path:
    quarantine_path = QUARANTINE_DIR / source_id / content_hash
    quarantine_path.mkdir(parents=True, exist_ok=True)
    quarantined_pdf = quarantine_path / "original.pdf"
    error_manifest = quarantine_path / "error.yaml"

    if not quarantined_pdf.exist():
        shutil.copy2(file_path, quarantined_pdf)

    error_data = {
        "source_id": source_id,
        "content_hash": content_hash,
        "source_path": str(file_path),
        "quarantined_path": str(quarantined_pdf),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "quarantined_at": datetime.now(UTC).isoformat(),
    }
    temporary_manifest = error_manifest.with_suffix(".yaml.tmp")
    with temporary_manifest.open("w", encoding="utf-8") as file:
        yaml.safe_dump(error_data, file, allow_unicode=True, sort_keys=False)
    temporary_manifest.replace(error_manifest)
    return quarantined_pdf

def write_manifest(
    manifest_path: Path,
    config_path: Path,
    config: dict[str, Any],
    source_path: Path,
    stored_path: Path,
    content_hash: str,
    page_count: int,
    size_bytes: int,
    ingestion_run_id: str,
) -> None:
    manifest = {
        "schema_version": "1.0",
        "raw_object_id": f"RAW-{content_hash}",
        "source_id": config["source"]["source_id"],
        "source_type": config["source"]["source_type"],
        "file": {
            "original_filename": config["file"]["original_filename"],
            "normalized_filename": config["file"]["normalized_filename"],
            "stored_fliename": stored_path.name,
            "media_type": config["file"]["media_type"],
            "language": config["file"]["langauge"],
            "size_bytes": size_bytes,
            "page_count": page_count,
            "sha256": content_hash,
        },
        "paths": {
            "source_path": str(source_path),
            "stored_path": str(stored_path),
            "config_path": str(config_path),
        },
        "ingestion": {
            "ingestion_run_id": ingestion_run_id,
            "ingested_at": datetime.now(UTC).isoformat(),
            "status": "SUCCESS",
        },
    }
    temporary_manifest = manifest_path.with_suffix(".yaml.tmp")
    # with_suffix: path 객체의 파일 확장자를 바꿔서 새로운 path를 만드는 메서드
    with temporary_manifest.open("w", encoding="utf-8") as file:
        yaml.safe_dump(manifest, file, allow_unicode=True, sort_keys=False)
    # safe_dump(value, stream)
    # value를 yaml 형식으로 변환해서 stream(파일)에 써라
    temporary_manifest.replace(manifest_path)

def ingest_source(yaml_path: Path, ingestion_run_id: str) -> str:
    try:
        config = load_yaml(yaml_path)
        source_id = config["source"]["source_id"]
        normalized_filename = config["file"]["normalized_filename"]
        file_path = resolve_input_path(config)
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        print(
            f"[CONFIG_ERROR] {yaml_path.name}: "
            f"{type(error).__name__}: {error}"
        )
        return "CONFIG_ERROR"

    if not file_path.is_file():
        print(f"[MISSING] {source_id}: {file_path}")
        return "MISSING"

    print(f"[SOURCE] {source_id}")
    print(f"[FILE] {normalized_filename}")
    print(f"[PATH] {file_path}")
    content_hash = sha256_file(file_path)

    try:
        page_count = inspect_pdf(file_path)
    except (fitz.FileDataError, ValueError, RuntimeError) as error:
        quarantine_path = quarantine_file(
            file_path=file_path,
            source_id=source_id,
            content_hash=content_hash,
            error=error,
        )
        print(
            f"[QUARANTINED] {source_id}: "
            f"{type(error).__name__}: {error}"
        )
        print(f"[QUARANTINE_PATH] {quarantine_path}")
        return "QUARANTINED"

    size_bytes = file_path.stat().st_size
    destination_dir = RAW_DATA_DIR / source_id / content_hash
    original_pdf_path = destination_dir / "original.pdf"
    manifest_path = destination_dir / "manifest.yaml"
    destination_dir.mkdir(parents=True, exist_ok=True)

    if original_pdf_path.is_file():
        stored_hash = sha256_file(original_pdf_path)
        if stored_hash == content_hash:
            if not manifest_path.is_file():
                write_manifest(
                    manifest_path=manifest_path,
                    config_path=yaml_path,
                    config=config,
                    source_path=file_path,
                    stored_path=original_pdf_path,
                    content_hash=content_hash,
                    page_count=page_count,
                    size_bytes=size_bytes,
                    ingestion_run_id=ingestion_run_id,
                )
            print(f"[DUPLICATE] existing Raw Object reused: {source_id}")
            return "DUPLICATE"
        raise RuntimeError(
            "Raw destination contains a different file: "
            f"path={original_pdf_path}, expected_hash={content_hash}, "
            f"stored_hash={stored_hash}"
        )

    temporary_pdf_path = destination_dir / "original.pdf.tmp"
    shutil.copy2(file_path, temporary_pdf_path)
    copied_hash = sha256_file(temporary_pdf_path)
    if copied_hash != content_hash:
        temporary_pdf_path.unlink(missing_ok=True)
        raise RuntimeError(
            "hash mismatch after copy: "
            f"source={content_hash}, copied={copied_hash}"            
        )
    temporary_pdf_path.replace(original_pdf_path)

    write_manifest(
        manifest_path=manifest_path,
        config_path=yaml_path,
        config=config,
        source_path=file_path,
        stored_path=original_pdf_path,
        content_hash=content_hash,
        page_count=page_count,
        size_bytes=size_bytes,
        ingestion_run_id=ingestion_run_id,
    )
    print(f"[STORED] {source_id}")
    print(f"[PAGES] {page_count}")
    print(f"[SHA256] {content_hash}")
    return "STORED"

def main() -> int:
    ingestion_run_id = (
        "INGEST-"
        + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    counters = {
        "DISCOVERED": 0,
        "VALIDATED": 0,
        "STORED": 0,
        "DUPLICATE": 0,
        "MISSING": 0,
        "QUARANTINED": 0,
        "CONFIG_ERROR": 0,
    }
    yaml_files = sorted(CONFIG_DIR.glob("*.yaml"))
    if not yaml_files:
        print(f"[ERROR] no source YAML files found: {CONFIG_DIR}")
        return 1

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    print("[LEGAL QA FACTORY - RAW INGESTION]")
    print(f"[RUN] {ingestion_run_id}")
    print(f"[CONFIG_DIR] {CONFIG_DIR}")
    print()

    for yaml_path in yaml_files:
        counters["DISCOVERED"] += 1
        status = ingest_source(yaml_path, ingestion_run_id)
        counters[status] += 1
        if status in {"STORED", "DUPLICATE"}:
            counters["VALIDATED"] += 1
        print()

    print("[INGESTION SUMMARY]")
    print(f"Run ID:                 {ingestion_run_id}")
    print(f"Sources discovered:     {counters['DISCOVERED']}")
    print(f"Raw objects validated:  {counters['VALIDATED']}")
    print(f"New objects stored:     {counters['STORED']}")
    print(f"Duplicates skipped:     {counters['DUPLICATE']}")
    print(f"Files quarantined:      {counters['QUARANTINED']}")
    print(f"Source files missing:   {counters['MISSING']}")
    print(f"Configuration errors:   {counters['CONFIG_ERROR']}")

    has_blocking_failure = any(
        counters[key] > 0
        for key in ("MISSING", "QUARANTINED", "CONFIG_ERROR")
    )
    if has_blocking_failure:
        print("[RESULT] FAILED")
        return 1
    print("[RESULT] PASSED")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())