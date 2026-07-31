from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from legal_qa_factory.blueprints.family_audit import (
    audit_family_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Family ranking errors without parent-group leakage."
    )
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = pq.read_table(args.training_data).to_pylist()
    excluded = [
        row
        for row in rows
        if not row.get("ranking_training_eligible", True)
    ]
    rows = [
        row
        for row in rows
        if row.get("ranking_training_eligible", True)
    ]
    report = audit_family_predictions(rows, k=args.k)
    report["excluded_row_count"] = len(excluded)
    report["exclusion_reason_counts"] = {
        reason: sum(
            row.get("ranking_exclusion_reason") == reason
            for row in excluded
        )
        for reason in sorted(
            {
                row.get("ranking_exclusion_reason")
                for row in excluded
                if row.get("ranking_exclusion_reason")
            }
        )
    }
    output = (
        args.output
        or args.training_data.parent / "family_confusion_audit.json"
    )
    pending = output.with_suffix(output.suffix + ".pending")
    pending.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pending.replace(output)
    print(f"[ROW_ACCURACY] {report['row_accuracy']}")
    print(f"[GROUP_ACCURACY] {report['group_accuracy']}")
    print(
        "[ERROR_TYPES] "
        + json.dumps(
            report["error_type_counts"],
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    print(
        "[VARIANTS] "
        + json.dumps(
            report["variant_summary"],
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    print(f"[OUTPUT] {output.resolve()}")


if __name__ == "__main__":
    main()
