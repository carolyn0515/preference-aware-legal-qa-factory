from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from legal_qa_factory.common.io import atomic_yaml_dump, load_yaml
from legal_qa_factory.lineage.evidence_alignment import align_claim
from legal_qa_factory.lineage.models import (
    CLAIM_EVIDENCE_SCHEMA,
    CLAIM_FEATURE_SCHEMA,
    QA_FLOW_SCHEMA,
)
from legal_qa_factory.lineage.quality import validate_lineage
from legal_qa_factory.lineage.trace_builder import (
    aggregate_flow_patterns,
    build_qa_flows,
)
from legal_qa_factory.retrieval.lexical import BM25Index
from legal_qa_factory.retrieval.traversal import build_node_indexes

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer deterministic Gold claim-to-legal-evidence lineage."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def load_legal_corpus() -> tuple[list[dict], list[dict], dict[str, list[str]], bool]:
    propositions, nodes = [], []
    functions_by_proposition: dict[str, list[str]] = {}
    prompt_ids, schema_generations = set(), set()
    for proposition_path in sorted(
        (ROOT / "data" / "silver").glob("*/*/propositions.parquet")
    ):
        document_nodes = pq.read_table(
            proposition_path.parent / "legal_nodes.parquet"
        ).to_pylist()
        document_nodes_by_id = {
            row["legal_node_id"]: row for row in document_nodes
        }
        document_propositions = pq.read_table(proposition_path).to_pylist()
        for proposition in document_propositions:
            node = document_nodes_by_id[proposition["legal_node_id"]]
            article = document_nodes_by_id[node["article_node_id"]]
            proposition["retrieval_text"] = " ".join(
                value
                for value in (
                    article["citation_label"],
                    article.get("title"),
                    proposition["text"],
                )
                if value
            )
        propositions.extend(document_propositions)
        nodes.extend(document_nodes)
        function_path = proposition_path.parent / "legal_functions.parquet"
        if function_path.exists():
            table = pq.read_table(function_path)
            schema_generations.add(
                "CURRENT"
                if "semantic_unit_type" in table.schema.names
                else "LEGACY"
            )
            for row in table.to_pylist():
                prompt_ids.add(row["prompt_id"])
                functions_by_proposition[row["proposition_id"]] = row["labels"]
    usable = len(prompt_ids) == 1 and len(schema_generations) == 1
    return propositions, nodes, functions_by_proposition, usable


def write_parquet(rows: list[dict], schema: pa.Schema, path: Path) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        pending,
        compression="zstd",
    )
    if pq.read_table(pending).num_rows != len(rows):
        raise RuntimeError(f"read-back count mismatch: {path}")
    pending.replace(path)


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    dataset_dir = args.dataset_dir.resolve()
    manifest = load_yaml(dataset_dir / "manifest.yaml")
    claims = pq.read_table(dataset_dir / "reference_claims.parquet").to_pylist()
    propositions, nodes, functions, function_usable = load_legal_corpus()
    nodes_by_id, _ = build_node_indexes(nodes)
    index = BM25Index(propositions, text_field="retrieval_text")

    features, candidates = [], []
    for claim in claims:
        feature, claim_candidates = align_claim(
            claim=claim,
            index=index,
            nodes_by_id=nodes_by_id,
            functions_by_proposition=functions,
            legal_function_usable=function_usable,
            top_k=args.top_k,
        )
        features.append(feature)
        candidates.extend(claim_candidates)
    validate_lineage(
        features,
        candidates,
        {row["proposition_id"] for row in propositions},
    )
    flows = build_qa_flows(features, candidates)
    flow_patterns = aggregate_flow_patterns(flows)
    artifact_dir = dataset_dir / "lineage"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(
        features,
        CLAIM_FEATURE_SCHEMA,
        artifact_dir / "claim_features.parquet",
    )
    write_parquet(
        candidates,
        CLAIM_EVIDENCE_SCHEMA,
        artifact_dir / "claim_evidence_candidates.parquet",
    )
    write_parquet(flows, QA_FLOW_SCHEMA, artifact_dir / "qa_flows.parquet")
    atomic_yaml_dump(flow_patterns, artifact_dir / "flow_patterns.yaml")

    relation_counts: dict[str, int] = defaultdict(int)
    for row in candidates:
        if row["selected"]:
            relation_counts[row["retrieval_relation"]] += 1
    atomic_yaml_dump(
        {
            "schema_version": "1.0",
            "lineage_method": "BM25_TREE_PATH_V1",
            "lineage_kind": "INFERRED",
            "truth_semantics": "CANDIDATE_EVIDENCE_ONLY",
            "input_sha256": manifest["input_sha256"],
            "claim_count": len(features),
            "candidate_count": len(candidates),
            "selected_candidate_count": sum(
                row["selected"] for row in candidates
            ),
            "qa_flow_count": len(flows),
            "legal_function_usable": function_usable,
            "legal_function_note": (
                None
                if function_usable
                else "Disabled because prompt/schema generations are mixed."
            ),
            "selected_relation_counts": dict(sorted(relation_counts.items())),
            "status": "PUBLISHED",
        },
        artifact_dir / "lineage_manifest.yaml",
    )
    print(
        f"[INFERRED] claims={len(features)} candidates={len(candidates)} "
        f"flows={len(flows)}"
    )
    print(f"[LEGAL_FUNCTION_USABLE] {function_usable}")
    print(f"[OUTPUT] {artifact_dir}")


if __name__ == "__main__":
    main()
