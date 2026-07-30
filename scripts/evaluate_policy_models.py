from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from legal_qa_factory.blueprints.evaluation import benchmark_policy_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Blueprint policy predictors with safety gates."
    )
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = pq.read_table(args.training_data).to_pylist()
    result = benchmark_policy_models(rows)
    output = args.output or args.training_data.parent / "model_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output)
    print(f"[EVALUATION] status={result['status']}")
    print(f"[SELECTION] {result['selection_status']}")
    print(f"[BEST_MODEL] {result['best_model_id']}")
    print(f"[DIAGNOSTIC_LEADER] {result['diagnostic_leader_id']}")
    if result["warning"]:
        print(f"[WARNING] {result['warning']}")
    print(f"[OUTPUT] {output.resolve()}")


if __name__ == "__main__":
    main()
