from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from legal_qa_factory.blueprints.compiler import compile_training_rows
from legal_qa_factory.blueprints.models import POLICY_TRAINING_SCHEMA
from legal_qa_factory.blueprints.validator import training_gate
from legal_qa_factory.common.io import atomic_yaml_dump


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile Gold QA lineage into policy-learning examples."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    qa_rows = pq.read_table(dataset_dir / "reference_qa.parquet").to_pylist()
    qa_flows = pq.read_table(dataset_dir / "lineage" / "qa_flows.parquet").to_pylist()
    tree_flows = pq.read_table(
        dataset_dir / "lineage" / "qa_tree_flows.parquet"
    ).to_pylist()
    rows, registry = compile_training_rows(qa_rows, qa_flows, tree_flows)
    gate = training_gate(rows)
    output_dir = dataset_dir / "policy"
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = output_dir / "policy_training.parquet.pending"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=POLICY_TRAINING_SCHEMA),
        pending,
        compression="zstd",
    )
    if pq.read_table(pending).num_rows != len(rows):
        raise RuntimeError("policy training read-back count mismatch")
    pending.replace(output_dir / "policy_training.parquet")
    atomic_yaml_dump(registry, output_dir / "pattern_registry.yaml")
    atomic_yaml_dump(gate, output_dir / "training_gate.yaml")
    print(
        f"[COMPILED] examples={len(rows)} patterns={len(registry['patterns'])}"
    )
    print(f"[TRAINING_GATE] {gate['status']}")
    if gate["blockers"]:
        print(f"[BLOCKERS] {','.join(gate['blockers'])}")
    print(f"[OUTPUT] {output_dir}")


if __name__ == "__main__":
    main()
