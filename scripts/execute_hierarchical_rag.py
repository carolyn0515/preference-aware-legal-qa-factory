from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from legal_qa_factory.blueprints.hierarchical import (
    recommend_hierarchical,
)
from legal_qa_factory.common.io import load_yaml
from legal_qa_factory.retrieval.blueprint_executor import (
    execute_blueprint_branches,
    load_silver_corpus,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute Top-2 Blueprints against the Silver legal tree."
    )
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/models/evidence_reranker.yaml",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_rows = pq.read_table(args.training_data).to_pylist()
    recommendation = recommend_hierarchical(
        args.question,
        policy_rows,
        k=5,
        top_families=2,
    )
    propositions, nodes = load_silver_corpus(ROOT)
    result = execute_blueprint_branches(
        question=args.question,
        blueprints=recommendation["blueprints"],
        propositions=propositions,
        nodes=nodes,
        config=load_yaml(args.config),
    )
    result["blueprint_recommendation"] = recommendation
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pending = args.output.with_suffix(args.output.suffix + ".pending")
        pending.write_text(serialized, encoding="utf-8")
        pending.replace(args.output)
        print(f"[OUTPUT] {args.output.resolve()}")
    print(serialized)


if __name__ == "__main__":
    main()

