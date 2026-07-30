from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from legal_qa_factory.blueprints.models import POLICY_TRAINING_SCHEMA
from legal_qa_factory.common.io import atomic_yaml_dump
from legal_qa_factory.review.importer import apply_reviews, read_review_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import completed lineage reviews.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    review_dir = dataset_dir / "review"
    policy_rows = pq.read_table(
        dataset_dir / "policy" / "policy_training.parquet"
    ).to_pylist()
    reviewed = apply_reviews(
        policy_rows,
        read_review_csv(review_dir / "answer_flow_review.csv"),
        read_review_csv(review_dir / "claim_evidence_review.csv"),
        read_review_csv(review_dir / "retrieval_flow_review.csv"),
    )
    output = dataset_dir / "policy" / "reviewed_policy_training.parquet"
    pending = output.with_suffix(output.suffix + ".pending")
    pq.write_table(
        pa.Table.from_pylist(reviewed, schema=POLICY_TRAINING_SCHEMA),
        pending,
        compression="zstd",
    )
    if pq.read_table(pending).num_rows != len(reviewed):
        raise RuntimeError("reviewed policy read-back count mismatch")
    pending.replace(output)
    eligible = sum(row["production_training_eligible"] for row in reviewed)
    atomic_yaml_dump(
        {
            "schema_version": "1.0",
            "status": "PUBLISHED",
            "label_provenance": "HUMAN_REVIEWED",
            "reviewed_count": len(reviewed),
            "production_eligible_count": eligible,
        },
        dataset_dir / "policy" / "reviewed_policy_manifest.yaml",
    )
    print(f"[IMPORTED] reviewed={len(reviewed)} production_eligible={eligible}")
    print(f"[OUTPUT] {output}")


if __name__ == "__main__":
    main()
