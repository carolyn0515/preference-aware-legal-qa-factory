from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from legal_qa_factory.silver.identifiers import proposition_id

SOURCE_NOTE = re.compile(
    r"<(?:개정|신설|삭제)[^>]*>"
    r"|\[(?:전문개정|본조신설|제목개정|종전)[^\]]*\]"
)
OPENING = {"(": ")", "[": "]", "{": "}", "「": "」", "『": "』", "“": "”", "‘": "’"}
CLOSING = set(OPENING.values())


@dataclass(frozen=True)
class Proposition:
    proposition_id: str
    legal_node_id: str
    article_node_id: str
    source_id: str
    source_version_hash: str
    proposition_sequence: int
    text: str
    char_start: int
    char_end: int
    split_rule: str
    bronze_record_ids: list[str]

    def as_dict(self) -> dict[str, Any]:
        return vars(self)


def _content_ranges(text: str) -> list[tuple[int, int]]:
    ranges = []
    start = 0
    for note in SOURCE_NOTE.finditer(text):
        if text[start : note.start()].strip():
            ranges.append((start, note.start()))
        start = note.end()
    if text[start:].strip():
        ranges.append((start, len(text)))
    return ranges


def _terminal_ends(text: str, start: int, end: int) -> list[int]:
    stack = []
    boundaries = []
    for index in range(start, end):
        character = text[index]
        if character in OPENING:
            stack.append(OPENING[character])
            continue
        if character in CLOSING:
            if stack and stack[-1] == character:
                stack.pop()
            continue
        if (
            character == "."
            and index > start
            and text[index - 1] == "다"
            and not stack
            and (index + 1 == end or text[index + 1].isspace())
        ):
            boundaries.append(index + 1)
    return boundaries


def split_propositions(node: dict[str, Any]) -> list[Proposition]:
    text = node["text"]
    if not text.strip():
        return []
    fragments: list[tuple[int, int, str]] = []
    for range_start, range_end in _content_ranges(text):
        cursor = range_start
        for end in _terminal_ends(text, range_start, range_end):
            fragments.append((cursor, end, "KOREAN_DECLARATIVE_TERMINAL"))
            cursor = end
        if text[cursor:range_end].strip():
            fragments.append((cursor, range_end, "RESIDUAL_CONTENT"))

    result = []
    for start, end, rule in fragments:
        left_trim = len(text[start:end]) - len(text[start:end].lstrip())
        right = text[start:end].rstrip()
        actual_start = start + left_trim
        actual_end = start + len(right)
        value = text[actual_start:actual_end]
        if not value:
            continue
        sequence = len(result) + 1
        result.append(
            Proposition(
                proposition_id=proposition_id(node["legal_node_id"], sequence, value),
                legal_node_id=node["legal_node_id"],
                article_node_id=node["article_node_id"],
                source_id=node["source_id"],
                source_version_hash=node["source_version_hash"],
                proposition_sequence=sequence,
                text=value,
                char_start=actual_start,
                char_end=actual_end,
                split_rule=rule,
                bronze_record_ids=node["bronze_record_ids"],
            )
        )
    return result
