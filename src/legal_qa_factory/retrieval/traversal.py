from __future__ import annotations

from collections import defaultdict
from typing import Any


def build_node_indexes(
    nodes: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_id = {node["legal_node_id"]: node for node in nodes}
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        if node.get("parent_node_id"):
            children[node["parent_node_id"]].append(node)
    for values in children.values():
        values.sort(key=lambda row: row["sequence"])
    return by_id, children


def relationship(
    origin_node: dict[str, Any], candidate_node: dict[str, Any]
) -> str:
    if origin_node["legal_node_id"] == candidate_node["legal_node_id"]:
        return "DIRECT"
    if origin_node.get("parent_node_id") == candidate_node["legal_node_id"]:
        return "PARENT"
    if candidate_node.get("parent_node_id") == origin_node["legal_node_id"]:
        return "CHILD"
    if (
        origin_node.get("parent_node_id")
        and origin_node.get("parent_node_id") == candidate_node.get("parent_node_id")
    ):
        return "SIBLING"
    if origin_node["article_node_id"] == candidate_node["article_node_id"]:
        return "SAME_ARTICLE"
    return "CROSS_NODE"


def node_path(
    node: dict[str, Any], nodes_by_id: dict[str, dict[str, Any]]
) -> list[str]:
    path = [node["node_type"]]
    parent_id = node.get("parent_node_id")
    while parent_id:
        parent = nodes_by_id[parent_id]
        path.append(parent["node_type"])
        parent_id = parent.get("parent_node_id")
    return list(reversed(path))
