from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from legal_qa_factory.common.io import atomic_yaml_dump, load_yaml
from legal_qa_factory.reference.profiler import (
    REFERENCE_FEATURE_SCHEMA,
    aggregate_profile,
    observed_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile observed structures in a processed Reference QA dataset."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    manifest = load_yaml(dataset_dir / "manifest.yaml")
    qa_rows = pq.read_table(dataset_dir / "reference_qa.parquet").to_pylist()
    claim_rows = pq.read_table(
        dataset_dir / "reference_claims.parquet"
    ).to_pylist()
    claims_by_qa: dict[str, list[dict]] = defaultdict(list)
    for claim in claim_rows:
        claims_by_qa[claim["reference_qa_id"]].append(claim)
    for claims in claims_by_qa.values():
        claims.sort(key=lambda row: row["claim_sequence"])

    features = [
        observed_features(row, claims_by_qa[row["reference_qa_id"]])
        for row in qa_rows
    ]
    profile = aggregate_profile(
        features, input_sha256=manifest["input_sha256"]
    )
    artifact_dir = dataset_dir / "analysis"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pending = artifact_dir / "observed_features.parquet.pending"
    pq.write_table(
        pa.Table.from_pylist(features, schema=REFERENCE_FEATURE_SCHEMA),
        pending,
        compression="zstd",
    )
    if pq.read_table(pending).num_rows != len(features):
        raise RuntimeError("Reference feature read-back count mismatch")
    pending.replace(artifact_dir / "observed_features.parquet")
    atomic_yaml_dump(profile, artifact_dir / "preference_profile.yaml")
    statuses = {
        item["status"] for item in profile["preference_candidates"]
    }
    print(f"[PROFILED] qa={len(features)} profiler={profile['profiler_id']}")
    print(f"[CANDIDATE_STATUSES] {','.join(sorted(statuses))}")
    print(f"[OUTPUT] {artifact_dir}")


if __name__ == "__main__":
    main()
