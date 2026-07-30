from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from legal_qa_factory.blueprints.ranking import _cosine
from legal_qa_factory.lineage.evidence_alignment import align_claim
from legal_qa_factory.lineage.hierarchy_expander import (
    RELATION_DECAY,
    build_article_indexes,
    expand_anchor,
    proposition_indexes,
)
from legal_qa_factory.retrieval.lexical import BM25Index
from legal_qa_factory.retrieval.query_analysis import lexical_terms
from legal_qa_factory.retrieval.traversal import build_node_indexes

ACTION_RELATIONS = {
    "EXPAND_PARENT": {"PARENT_CONTEXT"},
    "EXPAND_CHILDREN": {"CHILD_ENUMERATION"},
    "SEARCH_ROLE_SIBLINGS": {"SAME_ARTICLE_ROLE"},
    "FOLLOW_ARTICLE_REFERENCE": {"REFERENCED_ARTICLE"},
    "FOLLOW_DECREE_DELEGATION": {"IMPLEMENTING_DECREE"},
}


def load_silver_corpus(root: Path) -> tuple[list[dict], list[dict]]:
    propositions, nodes = [], []
    for proposition_path in sorted(
        (root / "data" / "silver").glob("*/*/propositions.parquet")
    ):
        document_nodes = pq.read_table(
            proposition_path.parent / "legal_nodes.parquet"
        ).to_pylist()
        nodes_by_id = {
            row["legal_node_id"]: row for row in document_nodes
        }
        document_propositions = pq.read_table(
            proposition_path
        ).to_pylist()
        for proposition in document_propositions:
            node = nodes_by_id[proposition["legal_node_id"]]
            article = nodes_by_id[node["article_node_id"]]
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
    if not propositions or not nodes:
        raise ValueError("Silver legal corpus is empty")
    return propositions, nodes


def _allowed_relations(actions: list[str]) -> set[str]:
    return {
        relation
        for action in actions
        for relation in ACTION_RELATIONS.get(action, set())
    }


def _score_evidence(
    *,
    question: str,
    text: str,
    question_relevance: float,
    hierarchy_proximity: float,
    weights: dict[str, float],
) -> dict[str, float]:
    question_terms = lexical_terms(question)
    evidence_terms = lexical_terms(text)
    query_set = set(question_terms)
    evidence_set = set(evidence_terms)
    coverage = (
        len(query_set & evidence_set) / len(query_set)
        if query_set
        else 0.0
    )
    lexical_similarity = _cosine(question_terms, evidence_terms)
    relevance = max(question_relevance, lexical_similarity)
    final_score = (
        weights["question_relevance"] * relevance
        + weights["query_coverage"] * coverage
        + weights["hierarchy_proximity"] * hierarchy_proximity
    )
    return {
        "question_relevance": round(relevance, 6),
        "query_coverage": round(coverage, 6),
        "hierarchy_proximity": round(hierarchy_proximity, 6),
        "final_score": round(final_score, 6),
    }


def execute_blueprint_branches(
    *,
    question: str,
    blueprints: list[dict[str, Any]],
    propositions: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    nodes_by_id, children_by_node = build_node_indexes(nodes)
    _, propositions_by_node = proposition_indexes(propositions)
    article_nodes, decree_index = build_article_indexes(nodes)
    index = BM25Index(propositions, text_field="retrieval_text")
    settings = config["execution"]
    _, anchors = align_claim(
        claim={
            "reference_claim_id": "LIVE-QUERY",
            "reference_qa_id": "LIVE-QUERY",
            "claim_sequence": 1,
            "text": question,
        },
        index=index,
        nodes_by_id=nodes_by_id,
        functions_by_proposition={},
        legal_function_usable=False,
        top_k=settings["anchor_k"],
    )
    branches = []
    for blueprint in blueprints:
        actions = blueprint["selected_retrieval_actions"]
        allowed_relations = _allowed_relations(actions)
        evidence_by_id: dict[str, dict[str, Any]] = {}
        for anchor in anchors:
            anchor_score = _score_evidence(
                question=question,
                text=anchor["evidence_text"],
                question_relevance=anchor["final_score"],
                hierarchy_proximity=1.0,
                weights=config["weights"],
            )
            evidence_by_id[anchor["evidence_proposition_id"]] = {
                "proposition_id": anchor["evidence_proposition_id"],
                "source_id": anchor["source_id"],
                "article_citation_label": anchor[
                    "article_citation_label"
                ],
                "citation_label": anchor["citation_label"],
                "text": anchor["evidence_text"],
                "relation": "DIRECT_LEXICAL",
                "reasons": ["BM25_ANCHOR"],
                "score_components": anchor_score,
            }
            if not allowed_relations:
                continue
            expansions = expand_anchor(
                anchor=anchor,
                answer_roles=blueprint["selected_answer_roles"],
                propositions_by_node=propositions_by_node,
                nodes_by_id=nodes_by_id,
                children_by_node=children_by_node,
                article_nodes=article_nodes,
                decree_reference_index=decree_index,
                limit=settings["expansion_limit_per_anchor"],
            )
            for expansion in expansions:
                if expansion["expansion_relation"] not in allowed_relations:
                    continue
                score = _score_evidence(
                    question=question,
                    text=expansion["context_text"],
                    question_relevance=expansion["expansion_score"],
                    hierarchy_proximity=RELATION_DECAY[
                        expansion["expansion_relation"]
                    ],
                    weights=config["weights"],
                )
                evidence = {
                    "proposition_id": expansion["context_proposition_id"],
                    "source_id": expansion["source_id"],
                    "article_citation_label": expansion[
                        "article_citation_label"
                    ],
                    "citation_label": expansion["citation_label"],
                    "text": expansion["context_text"],
                    "relation": expansion["expansion_relation"],
                    "reasons": expansion["expansion_reasons"],
                    "score_components": score,
                }
                previous = evidence_by_id.get(evidence["proposition_id"])
                if (
                    previous is None
                    or score["final_score"]
                    > previous["score_components"]["final_score"]
                ):
                    evidence_by_id[evidence["proposition_id"]] = evidence
        ranked = sorted(
            evidence_by_id.values(),
            key=lambda row: (
                -row["score_components"]["final_score"],
                row["proposition_id"],
            ),
        )[: settings["evidence_k_per_branch"]]
        top_n = ranked[: config["selection"]["branch_score_top_n"]]
        branch_score = (
            sum(row["score_components"]["final_score"] for row in top_n)
            / len(top_n)
            if top_n
            else 0.0
        )
        branches.append(
            {
                "pattern_family": blueprint["pattern_family"],
                "family_rank": blueprint["family_rank"],
                "family_score": blueprint["family_score"],
                "selected_retrieval_actions": actions,
                "selected_answer_roles": blueprint[
                    "selected_answer_roles"
                ],
                "branch_evidence_score": round(branch_score, 6),
                "evidence": ranked,
            }
        )
    branches.sort(
        key=lambda row: (
            -row["branch_evidence_score"],
            row["family_rank"],
        )
    )
    return {
        "method": config["reranker_id"],
        "selection_policy": config["selection"]["policy"],
        "question": question,
        "anchor_count": len(anchors),
        "selected_branch": branches[0] if branches else None,
        "branches": branches,
    }
