from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from legal_qa_factory.blueprints.ranking import evaluate_grouped_ranker
from legal_qa_factory.common.io import load_yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate graded Blueprint ranking without group leakage."
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
    rows = pq.read_table(args.training_data).to_pylist()
    config = load_yaml(args.config)
    results = [
        evaluate_grouped_ranker(rows, config, k=k)
        for k in config["prediction"]["k_values"]
    ]
    leader = max(
        results,
        key=lambda result: (
            result["metrics"].get("ndcg@5", 0.0),
            result["metrics"]["reciprocal_rank"],
        ),
    )
    report = {
        "schema_version": "1.0",
        "status": "DIAGNOSTIC_ONLY",
        "selection_metric": "ndcg@5",
        "diagnostic_leader_id": leader["model_id"],
        "relevance_config": config["relevance"],
        "models": results,
        "warning": (
            "Synthetic preference labels are not production ground truth."
        ),
    }
    output = args.training_data.parent / "ranking_evaluation.json"
    pending = output.with_suffix(output.suffix + ".pending")
    pending.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pending.replace(output)
    print("[EVALUATION] DIAGNOSTIC_ONLY")
    print(f"[LEADER] {leader['model_id']}")
    for result in results:
        print(
            f"[METRICS] {result['model_id']} "
            + json.dumps(
                result["metrics"], ensure_ascii=False, sort_keys=True
            )
        )
    print(f"[OUTPUT] {output.resolve()}")


if __name__ == "__main__":
    main()

