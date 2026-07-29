import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from legal_qa_factory.common.hashing import sha256_file
from legal_qa_factory.common.io import atomic_yaml_dump
from legal_qa_factory.silver.models import LegalNode
from legal_qa_factory.silver.schema import SILVER_SCHEMA


def publish(
    nodes: list[LegalNode],
    output_dir: Path,
    report: dict[str, object],
    exclusions: list[dict[str, object]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "legal_nodes.parquet.tmp"
    final = output_dir / "legal_nodes.parquet"
    table = pa.Table.from_pylist(
        [node.as_dict() for node in nodes], schema=SILVER_SCHEMA
    )
    pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
    if pq.read_table(temporary).num_rows != len(nodes):
        raise RuntimeError("Silver read-back verification failed")
    temporary.replace(final)
    with (output_dir / "excluded_boilerplate.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for exclusion in exclusions:
            file.write(json.dumps(exclusion, ensure_ascii=False) + "\n")
    atomic_yaml_dump(
        {
            "schema_version": "1.0",
            "dataset": "silver_legal_node",
            "source_id": nodes[0].source_id,
            "source_version_hash": nodes[0].source_version_hash,
            "output_sha256": sha256_file(final),
            "quality": report,
            "status": "PUBLISHED",
        },
        output_dir / "manifest.yaml",
    )
    return final
