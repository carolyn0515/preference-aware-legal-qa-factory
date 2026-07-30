from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from legal_qa_factory.blueprints.ranking import compile_ranking_rows
from legal_qa_factory.blueprints.ranking_models import (
    BLUEPRINT_RANKING_SCHEMA,
)
from legal_qa_factory.common.io import atomic_yaml_dump, load_yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile policy examples into graded ranking pairs."
    )
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/models/blueprint_ranker.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_rows = pq.read_table(args.training_data).to_pylist()
    config = load_yaml(args.config)
    ranking_rows, candidates = compile_ranking_rows(policy_rows, config)
    output = args.training_data.parent / "blueprint_ranking.parquet"
    pending = output.with_suffix(output.suffix + ".pending")
    pq.write_table(
        pa.Table.from_pylist(ranking_rows, schema=BLUEPRINT_RANKING_SCHEMA),
        pending,
        compression="zstd",
    )
    if pq.read_table(pending).num_rows != len(ranking_rows):
        raise RuntimeError("ranking dataset read-back count mismatch")
    pending.replace(output)
    atomic_yaml_dump(
        {
            "schema_version": "1.0",
            "target_semantics": "GRADED_RELEVANCE",
            "query_count": len(policy_rows),
            "candidate_count": len(candidates),
            "pair_count": len(ranking_rows),
            "relevance_config": config["relevance"],
            "status": "PUBLISHED",
        },
        output.parent / "blueprint_ranking_manifest.yaml",
    )
    print(
        f"[COMPILED] queries={len(policy_rows)} "
        f"candidates={len(candidates)} pairs={len(ranking_rows)}"
    )
    print("[TARGET] GRADED_RELEVANCE")
    print(f"[OUTPUT] {output.resolve()}")


if __name__ == "__main__":
    main()

