from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from legal_qa_factory.retrieval.query_analysis import lexical_terms

ARTICLE_REFERENCE = re.compile(r"제(?P<number>\d+)조(?:의(?P<branch>\d+))?")
ROLE_MARKERS = {
    "CONDITION": ("경우", "때", "요건", "사유"),
    "EXCEPTION_NOTICE": ("다만", "제외", "불구", "아니 된다"),
    "PROCEDURE": ("방법", "절차", "신청", "요청", "확인"),
    "SANCTION_NOTICE": ("과징금", "과태료", "벌금", "제재", "손해배상"),
}
RELATION_DECAY = {
    "PARENT_CONTEXT": 0.90,
    "CHILD_ENUMERATION": 0.95,
    "SAME_ARTICLE_ROLE": 0.82,
    "REFERENCED_ARTICLE": 0.78,
    "IMPLEMENTING_DECREE": 0.75,
}


def proposition_indexes(
    propositions: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    by_id = {row["proposition_id"]: row for row in propositions}
    by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for proposition in propositions:
        by_node[proposition["legal_node_id"]].append(proposition)
    for values in by_node.values():
        values.sort(key=lambda row: row["proposition_sequence"])
    return by_id, by_node


def _article_label(match: re.Match[str]) -> str:
    return f"제{match.group('number')}조" + (
        f"의{match.group('branch')}" if match.group("branch") else ""
    )


def _role_matches(text: str, answer_roles: list[str]) -> list[str]:
    return [
        role
        for role in answer_roles
        if role in ROLE_MARKERS
        and any(marker in text for marker in ROLE_MARKERS[role])
    ]


def _title_similarity(left: str, right: str) -> float:
    left_terms = set(lexical_terms(left))
    right_terms = set(lexical_terms(right))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / min(len(left_terms), len(right_terms))


def expand_anchor(
    *,
    anchor: dict[str, Any],
    answer_roles: list[str],
    propositions_by_node: dict[str, list[dict[str, Any]]],
    nodes_by_id: dict[str, dict[str, Any]],
    children_by_node: dict[str, list[dict[str, Any]]],
    article_nodes: dict[tuple[str, str], dict[str, Any]],
    decree_reference_index: dict[str, list[dict[str, Any]]],
    limit: int = 12,
) -> list[dict[str, Any]]:
    anchor_node = nodes_by_id[anchor["evidence_legal_node_id"]]
    anchor_article = nodes_by_id[anchor_node["article_node_id"]]
    proposals: list[tuple[str, dict[str, Any], list[str]]] = []
    delegation_reasons = []

    parent_id = anchor_node.get("parent_node_id")
    while parent_id:
        parent = nodes_by_id[parent_id]
        proposals.append(("PARENT_CONTEXT", parent, ["ANCESTOR_OF_ANCHOR"]))
        parent_id = parent.get("parent_node_id")

    for child in children_by_node.get(anchor_node["legal_node_id"], []):
        proposals.append(
            ("CHILD_ENUMERATION", child, ["DIRECT_CHILD_OF_ANCHOR"])
        )

    for sibling in children_by_node.get(anchor_article["legal_node_id"], []):
        if sibling["legal_node_id"] == anchor_node["legal_node_id"]:
            continue
        matched_roles = _role_matches(sibling["text"], answer_roles)
        if matched_roles:
            proposals.append(("SAME_ARTICLE_ROLE", sibling, matched_roles))
            if "대통령령" in sibling["text"]:
                delegation_reasons.append(
                    f"DELEGATION_FOUND_IN:{sibling['citation_label']}"
                )

    reference_text = " ".join(
        [anchor_node.get("text") or "", anchor_article.get("text") or ""]
    )
    for match in ARTICLE_REFERENCE.finditer(reference_text):
        label = _article_label(match)
        target = article_nodes.get((anchor["source_id"], label))
        if target and target["legal_node_id"] != anchor_article["legal_node_id"]:
            proposals.append(
                ("REFERENCED_ARTICLE", target, [f"EXPLICIT_REFERENCE:{label}"])
            )

    if "대통령령" in reference_text or delegation_reasons:
        targets = decree_reference_index.get(
            anchor_article["citation_label"], []
        )
        ranked = sorted(
            targets,
            key=lambda target: (
                -_title_similarity(
                    anchor_article.get("title") or "",
                    target.get("title") or "",
                ),
                target["citation_label"],
            ),
        )
        if ranked:
            best = _title_similarity(
                anchor_article.get("title") or "",
                ranked[0].get("title") or "",
            )
            ranked = [
                target
                for target in ranked
                if _title_similarity(
                    anchor_article.get("title") or "",
                    target.get("title") or "",
                )
                == best
            ]
        for target in ranked:
            proposals.append(
                (
                    "IMPLEMENTING_DECREE",
                    target,
                    [
                        "DELEGATION_MARKER:대통령령",
                        f"DECREE_REFERENCES:법 {anchor_article['citation_label']}",
                        *delegation_reasons,
                    ],
                )
            )

    expansions = []
    seen = {anchor["evidence_proposition_id"]}
    for relation, node, reasons in proposals:
        candidate_nodes = [node]
        if relation in {"REFERENCED_ARTICLE", "IMPLEMENTING_DECREE"}:
            candidate_nodes.extend(
                children_by_node.get(node["legal_node_id"], [])
            )
        for candidate_node in candidate_nodes:
            for proposition in propositions_by_node.get(
                candidate_node["legal_node_id"], []
            ):
                if proposition["proposition_id"] in seen:
                    continue
                seen.add(proposition["proposition_id"])
                expansions.append(
                    {
                        "reference_claim_id": anchor["reference_claim_id"],
                        "reference_qa_id": anchor["reference_qa_id"],
                        "anchor_proposition_id": anchor[
                            "evidence_proposition_id"
                        ],
                        "context_proposition_id": proposition["proposition_id"],
                        "context_legal_node_id": candidate_node[
                            "legal_node_id"
                        ],
                        "source_id": proposition["source_id"],
                        "article_citation_label": nodes_by_id[
                            candidate_node["article_node_id"]
                        ]["citation_label"],
                        "citation_label": candidate_node["citation_label"],
                        "node_type": candidate_node["node_type"],
                        "context_text": proposition["text"],
                        "expansion_relation": relation,
                        "expansion_reasons": reasons,
                        "anchor_score": anchor["final_score"],
                        "expansion_score": anchor["final_score"]
                        * RELATION_DECAY[relation],
                        "traversal_step": len(expansions) + 1,
                    }
                )
                if len(expansions) >= limit:
                    return expansions
    return expansions


def build_article_indexes(
    nodes: list[dict[str, Any]],
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    articles = {
        (node["source_id"], node["citation_label"]): node
        for node in nodes
        if node["node_type"] == "ARTICLE"
    }
    decree_references: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        if (
            node["source_type"] == "ENFORCEMENT_DECREE"
            and node["node_type"] == "ARTICLE"
        ):
            descendants = [
                value
                for value in nodes
                if value["article_node_id"] == node["legal_node_id"]
            ]
            text = " ".join(value.get("text") or "" for value in descendants)
            for match in re.finditer(r"법\s+제\d+조(?:의\d+)?", text):
                label = match.group().replace("법", "").strip()
                decree_references[label].append(node)
    return articles, decree_references
