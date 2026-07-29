from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from legal_qa_factory.common.hashing import sha256_file
from legal_qa_factory.common.io import atomic_yaml_dump
from legal_qa_factory.silver.semantics.propositions import split_propositions

PROPOSITION_SCHEMA = pa.schema(
    [
        pa.field("proposition_id", pa.string(), nullable=False),
        pa.field("legal_node_id", pa.string(), nullable=False),
        pa.field("article_node_id", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_version_hash", pa.string(), nullable=False),
        pa.field("proposition_sequence", pa.int32(), nullable=False),
        pa.field("text", pa.large_string(), nullable=False),
        pa.field("char_start", pa.int32(), nullable=False),
        pa.field("char_end", pa.int32(), nullable=False),
        pa.field("split_rule", pa.string(), nullable=False),
        pa.field("bronze_record_ids", pa.list_(pa.string()), nullable=False),
    ],
    metadata={
        b"schema_name": b"silver_legal_proposition",
        b"schema_version": b"1.0",
    },
)


def build_propositions(silver_root: Path) -> int:
    published = 0
    for node_path in sorted(silver_root.glob("*/*/legal_nodes.parquet")):
        nodes = pq.read_table(node_path).to_pylist()
        propositions = [
            proposition.as_dict()
            for node in nodes
            for proposition in split_propositions(node)
        ]
        if not propositions:
            raise ValueError(f"no propositions produced from {node_path}")
        ids = [item["proposition_id"] for item in propositions]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate proposition IDs: {node_path}")
        covered = {item["legal_node_id"] for item in propositions}
        expected = {node["legal_node_id"] for node in nodes if node["text"].strip()}
        if covered != expected:
            raise ValueError(
                f"proposition node coverage failure: {len(expected - covered)} missing"
            )
        output = node_path.parent / "propositions.parquet"
        temporary = output.with_suffix(".parquet.tmp")
        table = pa.Table.from_pylist(propositions, schema=PROPOSITION_SCHEMA)
        pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
        if pq.read_table(temporary).num_rows != len(propositions):
            raise RuntimeError("proposition Parquet read-back failure")
        temporary.replace(output)
        atomic_yaml_dump(
            {
                "schema_version": "1.0",
                "dataset": "silver_legal_proposition",
                "source_id": propositions[0]["source_id"],
                "source_version_hash": propositions[0]["source_version_hash"],
                "record_count": len(propositions),
                "covered_legal_node_count": len(covered),
                "eligible_legal_node_count": len(expected),
                "legal_node_coverage": len(covered) / len(expected),
                "output_sha256": sha256_file(output),
                "status": "PUBLISHED",
            },
            node_path.parent / "proposition_manifest.yaml",
        )
        print(f"[PUBLISHED] {len(propositions)} propositions -> {output}")
        published += 1
    return published
