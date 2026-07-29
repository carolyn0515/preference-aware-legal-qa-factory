from __future__ import annotations

from datetime import UTC, datetime

from legal_qa_factory.silver.identifiers import legal_node_id
from legal_qa_factory.silver.models import Candidate, LegalNode


def assemble(candidates: list[Candidate]) -> list[LegalNode]:
    nodes = []
    active: dict[str, LegalNode | None] = {
        "ARTICLE": None,
        "PARAGRAPH": None,
        "ITEM": None,
        "SUBITEM": None,
    }
    children = {"ARTICLE": 0, "PARAGRAPH": 0, "ITEM": 0, "SUBITEM": 0}
    now = datetime.now(UTC)
    for candidate in candidates:
        if candidate.region == "FRONT_MATTER":
            continue
        if candidate.node_type is None:
            leaf = next(
                (
                    active[x]
                    for x in ("SUBITEM", "ITEM", "PARAGRAPH", "ARTICLE")
                    if active[x]
                ),
                None,
            )
            if leaf:
                leaf.text = " ".join(filter(None, (leaf.text, candidate.body)))
                leaf.page_to = candidate.record["page_number"]
                leaf.bronze_record_ids.append(candidate.record["bronze_record_id"])
            continue
        kind = candidate.node_type
        if kind == "ARTICLE":
            parent = None
            path = f"{candidate.region}/{candidate.citation}"
        elif kind == "PARAGRAPH":
            parent = active["ARTICLE"]
            if parent is None:
                continue
            path = f"{parent.citation_label}/{candidate.marker}"
        elif kind == "ITEM":
            parent = active["PARAGRAPH"] or active["ARTICLE"]
            if parent is None:
                continue
            path = f"{parent.legal_node_id}/{candidate.marker}"
        else:
            parent = active["ITEM"]
            if parent is None:
                continue
            path = f"{parent.legal_node_id}/{candidate.marker}"
        article = active["ARTICLE"]
        node_id = legal_node_id(
            candidate.record["source_id"], candidate.record["content_hash"], path
        )
        children[kind] += 1
        node = LegalNode(
            node_id,
            candidate.record["source_id"],
            candidate.record["source_type"],
            candidate.record["content_hash"],
            kind,
            parent.legal_node_id if parent else None,
            node_id if kind == "ARTICLE" else article.legal_node_id,
            len(nodes) + 1,
            candidate.citation or candidate.marker,
            candidate.title,
            candidate.marker,
            candidate.body,
            candidate.region,
            candidate.record["page_number"],
            candidate.record["page_number"],
            [candidate.record["bronze_record_id"]],
            candidate.confidence,
            list(candidate.evidence),
            "korean_statute_structure_v1",
            now,
        )
        nodes.append(node)
        active[kind] = node
        order = ("ARTICLE", "PARAGRAPH", "ITEM", "SUBITEM")
        for descendant in order[order.index(kind) + 1 :]:
            active[descendant] = None
    return nodes
