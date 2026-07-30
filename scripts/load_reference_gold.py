from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from legal_qa_factory.common.io import atomic_yaml_dump
from legal_qa_factory.reference.claims import split_answer_claims
from legal_qa_factory.reference.loader import load_jsonl
from legal_qa_factory.reference.models import (
    REFERENCE_CLAIM_SCHEMA,
    REFERENCE_QA_SCHEMA,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and publish Reference/Gold QA as preference evidence."
    )
    parser.add_argument("--input", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, input_hash = load_jsonl(args.input)
    customer_id = records[0].customer_id
    reference_version = records[0].reference_version
    output_dir = (
        ROOT
        / "data"
        / "reference"
        / "processed"
        / customer_id
        / reference_version
        / input_hash
    )
    qa_rows = []
    claim_rows = []
    for record in records:
        row = asdict(record)
        row["observed_evidence_ids"] = list(record.observed_evidence_ids)
        row["metadata_json"] = json.dumps(
            record.metadata or {},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        row["input_sha256"] = input_hash
        row.pop("metadata")
        qa_rows.append(row)
        claim_rows.extend(
            split_answer_claims(record.reference_qa_id, record.answer)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pending_qa = output_dir / "reference_qa.parquet.pending"
    pending_claims = output_dir / "reference_claims.parquet.pending"
    pq.write_table(
        pa.Table.from_pylist(qa_rows, schema=REFERENCE_QA_SCHEMA),
        pending_qa,
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pylist(claim_rows, schema=REFERENCE_CLAIM_SCHEMA),
        pending_claims,
        compression="zstd",
    )
    if pq.read_table(pending_qa).num_rows != len(qa_rows):
        raise RuntimeError("Reference QA read-back count mismatch")
    if pq.read_table(pending_claims).num_rows != len(claim_rows):
        raise RuntimeError("Reference claim read-back count mismatch")
    pending_qa.replace(output_dir / "reference_qa.parquet")
    pending_claims.replace(output_dir / "reference_claims.parquet")
    atomic_yaml_dump(
        {
            "schema_version": "1.0",
            "dataset": "reference_qa",
            "truth_semantics": "PREFERENCE_EVIDENCE_ONLY",
            "customer_id": customer_id,
            "reference_version": reference_version,
            "input_path": str(args.input.resolve()),
            "input_sha256": input_hash,
            "qa_count": len(qa_rows),
            "claim_count": len(claim_rows),
            "source_kinds": sorted({row["source_kind"] for row in qa_rows}),
            "published_at": datetime.now(UTC).isoformat(),
            "status": "PUBLISHED",
        },
        output_dir / "manifest.yaml",
    )
    print(f"[PUBLISHED] qa={len(qa_rows)} claims={len(claim_rows)}")
    print("[SEMANTICS] PREFERENCE_EVIDENCE_ONLY")
    print(f"[OUTPUT] {output_dir}")


if __name__ == "__main__":
    main()
