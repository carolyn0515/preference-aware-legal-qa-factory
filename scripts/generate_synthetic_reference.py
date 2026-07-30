from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_qa_factory.common.hashing import sha256_file
from legal_qa_factory.common.io import atomic_yaml_dump
from legal_qa_factory.generation.expansion import (
    expand_rows,
    read_jsonl,
    read_plan,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand grounded seed QA under a versioned coverage plan."
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "configs/generation/synthetic_v3.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "examples/reference_gold.synthetic_v3.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = read_plan(args.plan)
    seed_path = ROOT / plan["seed"]["path"]
    result = expand_rows(read_jsonl(seed_path), plan)
    write_jsonl(args.output, result.rows)

    manifest = {
        **result.manifest,
        "plan_path": str(args.plan.resolve()),
        "plan_sha256": sha256_file(args.plan),
        "seed_path": str(seed_path.resolve()),
        "seed_sha256": sha256_file(seed_path),
        "output_path": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
    }
    manifest_path = args.output.with_suffix(".manifest.yaml")
    atomic_yaml_dump(manifest, manifest_path)

    print(
        "[GENERATED] "
        f"seeds={manifest['seed_count']} rows={manifest['row_count']} "
        f"groups={manifest['parent_group_count']}"
    )
    print(
        "[COVERAGE] "
        + json.dumps(
            {
                "question_types": manifest["question_type_counts"],
                "difficulties": manifest["difficulty_counts"],
                "topics": len(manifest["topic_counts"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    print("[SPLIT_POLICY] GROUP_BY_PARENT_EXAMPLE_ID")
    print(f"[OUTPUT] {args.output.resolve()}")
    print(f"[MANIFEST] {manifest_path.resolve()}")


if __name__ == "__main__":
    main()

