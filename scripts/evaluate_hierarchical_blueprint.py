from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from legal_qa_factory.blueprints.hierarchical import (
    evaluate_hierarchical_ranker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate hierarchical family/component Blueprint ranking."
    )
    parser.add_argument("--training-data", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = pq.read_table(args.training_data).to_pylist()
    results = [
        evaluate_hierarchical_ranker(rows, k=k) for k in (1, 3, 5)
    ]
    leader = max(
        results,
        key=lambda result: (
            result["metrics"]["family_ndcg@3"],
            result["metrics"]["family_mrr"],
        ),
    )
    report = {
        "schema_version": "1.0",
        "status": "DIAGNOSTIC_ONLY",
        "target_design": (
            "FAMILY_RANKING_PLUS_COMPONENT_RANKING"
        ),
        "diagnostic_leader_id": leader["model_id"],
        "models": results,
    }
    output = args.training_data.parent / "hierarchical_evaluation.json"
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

