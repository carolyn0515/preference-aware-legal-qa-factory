from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from legal_qa_factory.bronze.schema import BRONZE_SCHEMA
from legal_qa_factory.common.hashing import sha256_file
from legal_qa_factory.common.io import atomic_yaml_dump


def publish_parquet(
    records: list[dict[str, Any]],
    output_dir: Path,
    manifest: dict[str, Any],
    run_id: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "records.parquet"
    temporary = output_dir / "records.parquet.tmp"
    table = pa.Table.from_pylist(records, schema=BRONZE_SCHEMA)
    # 여기서 arrow table로 변환
    pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
    # zstd = Zstandard 압축 
    # 같은 값이 많으면 그냥 저장하지 않고 인코딩같은거 함 -> parquet의 대표적인 최적화
    readback = pq.read_table(temporary)
    # 방금 저장한 파일 다시 읽음 -> 정상 저장 check
    if readback.num_rows != len(records) or not readback.schema.equals(BRONZE_SCHEMA):
        temporary.unlink(missing_ok=True)
        # tmp 삭제
        raise RuntimeError("Parquet read-back verification failed")
    temporary.replace(final_path)
    # Atomic Replace _ no 중간 상태
    atomic_yaml_dump(
        {
            "schema_version": "1.0",
            "dataset": "bronze_pdf_block",
            "source_id": manifest["source_id"],
            "raw_object_id": manifest["raw_object_id"],
            "source_content_hash": manifest["file"]["sha256"],
            "record_count": len(records),
            "output_sha256": sha256_file(final_path),
            "extraction_run_id": run_id,
            "status": "PUBLISHED",
        },
        output_dir / "manifest.yaml",
    )
    return final_path
