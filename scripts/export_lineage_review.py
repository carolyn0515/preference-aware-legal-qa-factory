from __future__ import annotations

import argparse
from pathlib import Path

from legal_qa_factory.common.io import atomic_yaml_dump, load_yaml
from legal_qa_factory.review.exporter import export_review_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export lineage review CSV files.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    source_manifest = load_yaml(dataset_dir / "manifest.yaml")
    lineage_manifest = load_yaml(
        dataset_dir / "lineage" / "lineage_manifest.yaml"
    )
    output_dir = dataset_dir / "review"
    counts = export_review_batch(dataset_dir, output_dir)
    atomic_yaml_dump(
        {
            "schema_version": "1.0",
            "status": "AWAITING_REVIEW",
            "input_sha256": source_manifest["input_sha256"],
            "lineage_method": lineage_manifest["lineage_method"],
            **counts,
            "instructions": {
                "do_not_edit": "columns before system_row_sha256",
                "answer_decisions": ["CORRECT", "CHANGE", "REMOVE"],
                "evidence_decisions": [
                    "CORRECT",
                    "PARTIAL",
                    "IRRELEVANT",
                    "NO_EVIDENCE_APPROPRIATE",
                    "MISSING_EVIDENCE",
                ],
                "retrieval_decisions": [
                    "REQUIRED",
                    "OPTIONAL",
                    "UNNECESSARY",
                ],
            },
        },
        output_dir / "review_manifest.yaml",
    )
    print(
        "[EXPORTED] "
        f"answer={counts['answer_review_count']} "
        f"evidence={counts['evidence_review_count']} "
        f"retrieval={counts['retrieval_review_count']}"
    )
    print(f"[OUTPUT] {output_dir}")


if __name__ == "__main__":
    main()
