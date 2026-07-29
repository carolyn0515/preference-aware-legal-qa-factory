from __future__ import annotations

from collections import defaultdict
from typing import Any

from legal_qa_factory.silver.models import LegalNode

CIRCLED = dict(zip("①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳", range(1, 21), strict=True))


def _number(node: LegalNode) -> int | None:
    if node.node_type == "PARAGRAPH":
        return CIRCLED.get(node.marker)
    if node.node_type == "ITEM" and node.marker.isdigit():
        return int(node.marker)
    return None


def audit_structure(nodes: list[LegalNode]) -> dict[str, Any]:
    children: dict[tuple[str | None, str], list[LegalNode]] = defaultdict(list)
    for node in nodes:
        children[(node.parent_node_id, node.node_type)].append(node)

    anomalies = []
    for (parent_id, node_type), siblings in children.items():
        numbered = [(node, _number(node)) for node in siblings]
        numbered = [(node, number) for node, number in numbered if number is not None]
        seen = set()
        previous = None
        for node, number in numbered:
            if number in seen:
                anomalies.append(
                    {
                        "anomaly_type": "DUPLICATE_SIBLING_MARKER",
                        "legal_node_id": node.legal_node_id,
                        "parent_node_id": parent_id,
                        "node_type": node_type,
                        "marker": node.marker,
                        "page": node.page_from,
                    }
                )
            if previous is not None and number > previous + 1:
                anomalies.append(
                    {
                        "anomaly_type": "NUMBERING_GAP",
                        "legal_node_id": node.legal_node_id,
                        "parent_node_id": parent_id,
                        "node_type": node_type,
                        "previous": previous,
                        "current": number,
                        "page": node.page_from,
                    }
                )
            seen.add(number)
            previous = number

    suspicious = []
    for node in nodes:
        reasons = []
        if node.boundary_confidence < 0.75:
            reasons.append("LOW_BOUNDARY_CONFIDENCE")
        if not node.text and node.node_type not in {"ARTICLE"}:
            reasons.append("EMPTY_STRUCTURAL_NODE")
        if "법제처" in node.text or "국가법령정보센터" in node.text:
            reasons.append("BOILERPLATE_LEAKAGE")
        if reasons:
            suspicious.append(
                {
                    "legal_node_id": node.legal_node_id,
                    "node_type": node.node_type,
                    "citation_label": node.citation_label,
                    "page_from": node.page_from,
                    "reasons": reasons,
                    "text_preview": node.text[:200],
                }
            )
    return {
        "anomaly_count": len(anomalies),
        "suspicious_node_count": len(suspicious),
        "numbering_anomalies": anomalies,
        "suspicious_nodes": suspicious,
    }
