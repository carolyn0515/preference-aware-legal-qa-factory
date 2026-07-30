from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from legal_qa_factory.blueprints.hierarchical import (
    recommend_hierarchical,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recommend Top-N hierarchical Retrieval Blueprints."
    )
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--top-families", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = pq.read_table(args.training_data).to_pylist()
    result = recommend_hierarchical(
        args.question,
        rows,
        k=args.k,
        top_families=args.top_families,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

