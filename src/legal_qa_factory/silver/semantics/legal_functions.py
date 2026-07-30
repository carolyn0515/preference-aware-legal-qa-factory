from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

import pyarrow as pa

LABELS = (
    "DEFINITION",
    "SCOPE",
    "OBLIGATION",
    "PROHIBITION",
    "PERMISSION",
    "CONDITION",
    "EXCEPTION",
    "PROCESS",
    "AUTHORITY",
    "SANCTION",
    "REFERENCE",
    "DELEGATION",
    "UNCLASSIFIED",
)

LEGAL_FUNCTION_SCHEMA = pa.schema(
    [
        pa.field("proposition_id", pa.string(), nullable=False),
        pa.field("legal_node_id", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_version_hash", pa.string(), nullable=False),
        pa.field("subject", pa.string()),
        pa.field("action", pa.string()),
        pa.field("object", pa.string()),
        pa.field("modality", pa.string(), nullable=False),
        pa.field("conditions", pa.list_(pa.string()), nullable=False),
        pa.field("exceptions", pa.list_(pa.string()), nullable=False),
        pa.field("labels", pa.list_(pa.string()), nullable=False),
        pa.field("evidence_phrases", pa.list_(pa.string()), nullable=False),
        pa.field("confidence", pa.float32(), nullable=False),
        pa.field("semantic_unit_type", pa.string(), nullable=False),
        pa.field("intrinsic_labels", pa.list_(pa.string()), nullable=False),
        pa.field("inherited_labels", pa.list_(pa.string()), nullable=False),
        pa.field("classification_method", pa.string(), nullable=False),
        pa.field("provider", pa.string(), nullable=False),
        pa.field("model", pa.string(), nullable=False),
        pa.field("prompt_id", pa.string(), nullable=False),
        pa.field("request_hash", pa.string(), nullable=False),
    ],
    metadata={
        b"schema_name": b"silver_legal_function",
        b"schema_version": b"1.0",
    },
)

INHERITABLE_FROM_ENUMERATION = frozenset(
    {"DEFINITION", "SCOPE", "CONDITION", "EXCEPTION"}
)
ENUMERATION_MARKERS = ("다음 각 호", "다음 각 목", "각 호의 어느 하나")

RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "proposition_id": {"type": "string"},
                    "subject": {"type": ["string", "null"]},
                    "action": {"type": ["string", "null"]},
                    "object": {"type": ["string", "null"]},
                    "modality": {
                        "type": "string",
                        "enum": ["MUST", "MUST_NOT", "MAY", "POWER", "NONE"],
                    },
                    "conditions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "exceptions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(LABELS)},
                    },
                    "evidence_phrases": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
                "required": [
                    "proposition_id",
                    "subject",
                    "action",
                    "object",
                    "modality",
                    "conditions",
                    "exceptions",
                    "labels",
                    "evidence_phrases",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


def build_payload(
    propositions: list[dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    items = []
    for proposition in propositions:
        node = nodes_by_id[proposition["legal_node_id"]]
        article = nodes_by_id.get(node["article_node_id"], node)
        parent = (
            nodes_by_id.get(node.get("parent_node_id"))
            if node.get("parent_node_id")
            else None
        )
        items.append(
            {
                "proposition_id": proposition["proposition_id"],
                "document_type": node["source_type"],
                "article_node_id": node["article_node_id"],
                "citation_label": node["citation_label"],
                "article_title": article["title"],
                "node_type": node["node_type"],
                "semantic_unit_type": semantic_unit_type(node, proposition["text"]),
                "ancestor_path": ancestor_path(node, nodes_by_id),
                "parent_lead_in": parent["text"][:500] if parent else None,
                "proposition": proposition["text"],
            }
        )
    return {"propositions": items}


def ancestor_path(
    node: dict[str, Any],
    nodes_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    path = [node["citation_label"]]
    parent_id = node.get("parent_node_id")
    while parent_id:
        parent = nodes_by_id[parent_id]
        path.append(parent["citation_label"])
        parent_id = parent.get("parent_node_id")
    return list(reversed(path))


def semantic_unit_type(node: dict[str, Any], text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("삭제<") or stripped == "삭제":
        return "DELETED"
    if node["node_type"] == "ARTICLE" and node["title"] == "목적":
        return "PURPOSE"
    if any(marker in stripped for marker in ENUMERATION_MARKERS):
        return "ENUMERATION_LEAD"
    if node["node_type"] in {"ITEM", "SUBITEM"} and not stripped.endswith("."):
        return "LIST_FRAGMENT"
    return "FULL_PROPOSITION"


def apply_label_inheritance(
    records: list[dict[str, Any]],
    propositions_by_id: dict[str, dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    lead_labels: dict[str, set[str]] = {}
    for record in records:
        proposition = propositions_by_id[record["proposition_id"]]
        node = nodes_by_id[proposition["legal_node_id"]]
        unit_type = semantic_unit_type(node, proposition["text"])
        record["semantic_unit_type"] = unit_type
        record["intrinsic_labels"] = list(record["labels"])
        if unit_type == "ENUMERATION_LEAD":
            lead_labels.setdefault(node["legal_node_id"], set()).update(
                label
                for label in record["intrinsic_labels"]
                if label in INHERITABLE_FROM_ENUMERATION
            )

    for record in records:
        proposition = propositions_by_id[record["proposition_id"]]
        node = nodes_by_id[proposition["legal_node_id"]]
        inherited: list[str] = []
        if record["semantic_unit_type"] == "LIST_FRAGMENT":
            inherited = sorted(
                lead_labels.get(node["parent_node_id"], set())
                - set(record["intrinsic_labels"])
            )
        record["inherited_labels"] = inherited
        effective = set(record["intrinsic_labels"]) | set(inherited)
        if len(effective) > 1:
            effective.discard("UNCLASSIFIED")
        record["labels"] = sorted(effective)
    return records


def request_hash(
    *,
    namespace: str,
    model: str,
    prompt_id: str,
    payload: dict[str, Any],
) -> str:
    canonical = json.dumps(
        {
            "namespace": namespace,
            "model": model,
            "prompt_id": prompt_id,
            "payload": payload,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def validate_result(
    response: dict[str, Any],
    propositions: list[dict[str, Any]],
) -> None:
    expected = [item["proposition_id"] for item in propositions]
    actual = [item["proposition_id"] for item in response["results"]]
    if actual != expected:
        raise ValueError("LLM result IDs or ordering do not match request")
    texts = {item["proposition_id"]: item["text"] for item in propositions}
    for result in response["results"]:
        if not result["labels"]:
            raise ValueError(f"empty labels: {result['proposition_id']}")
        if len(result["labels"]) != len(set(result["labels"])):
            raise ValueError(f"duplicate labels: {result['proposition_id']}")
        if "UNCLASSIFIED" in result["labels"] and result["labels"] != ["UNCLASSIFIED"]:
            raise ValueError(
                f"UNCLASSIFIED must be exclusive: {result['proposition_id']}"
            )
        source = texts[result["proposition_id"]]
        aligned = []
        missing = []
        for phrase in result["evidence_phrases"]:
            exact = align_evidence_phrase(source, phrase)
            if exact is None:
                missing.append(phrase)
            else:
                aligned.append(exact)
        if missing:
            raise ValueError(
                f"evidence phrases absent from source: {result['proposition_id']}"
            )
        result["evidence_phrases"] = aligned


def align_evidence_phrase(source: str, phrase: str) -> str | None:
    if phrase in source:
        return phrase

    source_compact = []
    source_positions = []
    for index, character in enumerate(source):
        if character.isspace():
            continue
        source_compact.append(character)
        source_positions.append(index)

    phrase_compact = "".join(
        character for character in phrase if not character.isspace()
    )
    if not phrase_compact:
        return None

    compact_source = "".join(source_compact)
    compact_start = compact_source.find(phrase_compact)
    if compact_start < 0:
        return None

    compact_end = compact_start + len(phrase_compact) - 1
    source_start = source_positions[compact_start]
    source_end = source_positions[compact_end] + 1
    return source[source_start:source_end]
