from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
SILVER_ROOT = ROOT / "data" / "silver"
OUTPUT_ROOT = ROOT / "data" / "artifacts" / "parser_audits"
INDENT = {"ARTICLE": 0, "PARAGRAPH": 1, "ITEM": 2, "SUBITEM": 3}


def tree_line(record: dict[str, Any]) -> str:
    depth = INDENT[record["node_type"]]
    title = f" ({record['title']})" if record["title"] else ""
    preview = " ".join(record["text"].split())[:100]
    evidence = ", ".join(record["boundary_evidence"])
    return (
        f"{'    ' * depth}├─ {record['node_type']} "
        f"{record['citation_label']}{title}\n"
        f"{'    ' * (depth + 1)}text: {preview}\n"
        f"{'    ' * (depth + 1)}page: {record['page_from']}"
        f"-{record['page_to']} | confidence: "
        f"{record['boundary_confidence']:.2f} | evidence: {evidence}\n"
        f"{'    ' * (depth + 1)}bronze: "
        f"{', '.join(record['bronze_record_ids'])}"
    )


def publish_preview(parquet_path: Path) -> Path:
    records = pq.read_table(parquet_path).to_pylist()
    source_id = records[0]["source_id"]
    output_dir = OUTPUT_ROOT / source_id
    output_dir.mkdir(parents=True, exist_ok=True)

    node_counts = dict(Counter(record["node_type"] for record in records))
    summary = {
        "source_id": source_id,
        "source_version_hash": records[0]["source_version_hash"],
        "record_count": len(records),
        "node_counts": node_counts,
        "region_counts": dict(Counter(record["region"] for record in records)),
        "bronze_lineage_coverage": (
            sum(bool(record["bronze_record_ids"]) for record in records) / len(records)
        ),
        "mean_boundary_confidence": (
            sum(record["boundary_confidence"] for record in records) / len(records)
        ),
        "inline_paragraph_count": sum(
            "INLINE_AFTER_ARTICLE" in record["boundary_evidence"] for record in records
        ),
        "boilerplate_leakage_count": sum(
            "법제처" in record["text"] or "국가법령정보센터" in record["text"]
            for record in records
        ),
    }
    with (output_dir / "quality_summary.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(summary, file, allow_unicode=True, sort_keys=False)

    with (output_dir / "tree_preview.txt").open("w", encoding="utf-8") as file:
        file.write(f"{source_id} STRUCTURE TREE\n")
        file.write("=" * 80 + "\n\n")
        file.write("\n\n".join(tree_line(record) for record in records))
        file.write("\n")

    with (output_dir / "sample_nodes.jsonl").open("w", encoding="utf-8") as file:
        for record in records[:50]:
            file.write(
                json.dumps(record, ensure_ascii=False, default=str, sort_keys=True)
                + "\n"
            )

    return output_dir


def main() -> None:
    paths = sorted(SILVER_ROOT.glob("*/*/legal_nodes.parquet"))
    if not paths:
        raise ValueError(f"no Silver datasets found under {SILVER_ROOT}")
    for path in paths:
        output = publish_preview(path)
        print(f"[PUBLISHED] Silver inspection artifacts -> {output}")


if __name__ == "__main__":
    main()
