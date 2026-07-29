from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from legal_qa_factory.silver.models import LegalNode
from legal_qa_factory.silver.structure.audit import audit_structure

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    paths = sorted((ROOT / "data" / "silver").glob("*/*/legal_nodes.parquet"))
    for path in paths:
        rows = pq.read_table(path).to_pylist()
        nodes = [LegalNode(**row) for row in rows]
        report = audit_structure(nodes)
        output = ROOT / "data" / "artifacts" / "parser_audits" / nodes[0].source_id
        output.mkdir(parents=True, exist_ok=True)
        for name, key in (
            ("numbering_anomalies.jsonl", "numbering_anomalies"),
            ("suspicious_nodes.jsonl", "suspicious_nodes"),
        ):
            with (output / name).open("w", encoding="utf-8") as file:
                for item in report[key]:
                    file.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(
            f"[AUDIT] {nodes[0].source_id}: "
            f"{report['anomaly_count']} numbering anomalies, "
            f"{report['suspicious_node_count']} suspicious nodes"
        )


if __name__ == "__main__":
    main()
