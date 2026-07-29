from pathlib import Path

import pyarrow.parquet as pq

from legal_qa_factory.silver.quality import validate
from legal_qa_factory.silver.structure.assembler import assemble
from legal_qa_factory.silver.structure.audit import audit_structure
from legal_qa_factory.silver.structure.boilerplate import remove_repeated_boilerplate
from legal_qa_factory.silver.structure.classifier import classify
from legal_qa_factory.silver.writer import publish

SUPPORTED = {"STATUTE", "ENFORCEMENT_DECREE"}


def build_silver_structure(bronze_root: Path, silver_root: Path) -> int:
    published = 0
    for path in sorted(bronze_root.glob("*/*/records.parquet")):
        records = sorted(
            pq.read_table(path).to_pylist(),
            key=lambda x: (x["page_number"], x["block_index"]),
        )
        if not records or records[0]["source_type"] not in SUPPORTED:
            continue
        cleaned, exclusions = remove_repeated_boilerplate(records)
        nodes = assemble(classify(cleaned))
        report = validate(nodes)
        audit = audit_structure(nodes)
        report["excluded_boilerplate_count"] = len(exclusions)
        report["numbering_anomaly_count"] = audit["anomaly_count"]
        report["suspicious_node_count"] = audit["suspicious_node_count"]
        destination = silver_root / nodes[0].source_id / nodes[0].source_version_hash
        output = publish(nodes, destination, report, exclusions)
        print(f"[PUBLISHED] {len(nodes)} nodes -> {output}")
        print(f"[QUALITY] {report}")
        published += 1
    if not published:
        raise ValueError("no supported Bronze datasets")
    return published
