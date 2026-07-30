from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from legal_qa_factory.blueprints.registry import recommend_knn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recommend an experimental Blueprint for a new question."
    )
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = pq.read_table(args.training_data).to_pylist()
    result = recommend_knn(args.question, rows, k=args.k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
