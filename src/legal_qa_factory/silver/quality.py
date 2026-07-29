from collections import Counter

from legal_qa_factory.silver.models import LegalNode


def validate(nodes: list[LegalNode]) -> dict[str, object]:
    if not nodes:
        raise ValueError("no Silver nodes")
    ids = {node.legal_node_id for node in nodes}
    if len(ids) != len(nodes):
        raise ValueError("duplicate Silver IDs")
    broken = [
        node.legal_node_id
        for node in nodes
        if node.parent_node_id and node.parent_node_id not in ids
    ]
    if broken:
        raise ValueError(f"broken parent lineage: {broken[:5]}")
    counts = dict(Counter(node.node_type for node in nodes))
    if not counts.get("ARTICLE"):
        raise ValueError("no articles detected")
    return {
        "record_count": len(nodes),
        "node_counts": counts,
        "bronze_lineage_coverage": sum(bool(x.bronze_record_ids) for x in nodes)
        / len(nodes),
        "mean_boundary_confidence": sum(x.boundary_confidence for x in nodes)
        / len(nodes),
    }
