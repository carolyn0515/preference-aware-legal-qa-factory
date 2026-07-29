from __future__ import annotations

from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    paths = sorted((ROOT / "data" / "silver").glob("*/*/propositions.parquet"))
    for path in paths:
        rows = pq.read_table(path).to_pylist()
        source_id = rows[0]["source_id"]
        output_dir = ROOT / "data" / "artifacts" / "parser_audits" / source_id
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "proposition_preview.txt").open(
            "w", encoding="utf-8"
        ) as file:
            for row in rows[:100]:
                file.write(
                    f"[{row['proposition_id']}]\n"
                    f"node={row['legal_node_id']} "
                    f"sequence={row['proposition_sequence']} "
                    f"offset={row['char_start']}:{row['char_end']} "
                    f"rule={row['split_rule']}\n"
                    f"{row['text']}\n\n"
                )
        summary = {
            "source_id": source_id,
            "proposition_count": len(rows),
            "covered_legal_node_count": len({row["legal_node_id"] for row in rows}),
            "split_rule_counts": dict(Counter(row["split_rule"] for row in rows)),
            "maximum_character_count": max(len(row["text"]) for row in rows),
            "mean_character_count": (sum(len(row["text"]) for row in rows) / len(rows)),
        }
        with (output_dir / "proposition_summary.yaml").open(
            "w", encoding="utf-8"
        ) as file:
            yaml.safe_dump(summary, file, allow_unicode=True, sort_keys=False)
        print(f"[PUBLISHED] proposition inspection -> {output_dir}")


if __name__ == "__main__":
    main()
