from __future__ import annotations

from typing import Any

from legal_qa_factory.silver.models import Candidate
from legal_qa_factory.silver.structure.patterns import (
    ADDENDUM,
    lexical_match,
    split_inline_branched_items,
)
from legal_qa_factory.silver.structure.typography import (
    body_font_size,
    heading_evidence,
)


def classify(records: list[dict[str, Any]]) -> list[Candidate]:
    normal_size = body_font_size(records)
    region = "FRONT_MATTER"
    result = []
    for record in records:
        text = record["normalized_text"]
        if ADDENDUM.match(text):
            region = "ADDENDUM"
        matched = lexical_match(text)
        if matched and matched[0] == "ARTICLE" and region == "FRONT_MATTER":
            region = "BODY"
        if not matched:
            result.append(
                Candidate(record, None, None, None, None, text, 0.0, (), region)
            )
            continue
        kind, marker, citation, title, body = matched
        if kind == "ARTICLE":
            heading = text[: len(text) - len(body)].strip()
            confidence, evidence = heading_evidence(record, heading, normal_size)
            if confidence < 0.75:
                result.append(
                    Candidate(
                        record,
                        None,
                        None,
                        None,
                        None,
                        text,
                        confidence,
                        evidence,
                        region,
                    )
                )
                continue
            inline = lexical_match(body)
            article_body = "" if inline and inline[0] == "PARAGRAPH" else body
            result.append(
                Candidate(
                    record,
                    kind,
                    marker,
                    citation,
                    title,
                    article_body.strip(),
                    confidence,
                    evidence,
                    region,
                )
            )
            if inline and inline[0] == "PARAGRAPH":
                inline_kind, inline_marker, inline_citation, _, inline_body = inline
                result.append(
                    Candidate(
                        record,
                        inline_kind,
                        inline_marker,
                        inline_citation,
                        None,
                        inline_body.strip(),
                        0.90,
                        ("LEXICAL_PREFIX", "INLINE_AFTER_ARTICLE"),
                        region,
                    )
                )
            continue
        else:
            confidence, evidence = 0.80, ("LEXICAL_PREFIX", "BLOCK_START")
        branches = []
        if kind == "ITEM":
            body, branches = split_inline_branched_items(body)
        result.append(
            Candidate(
                record,
                kind,
                marker,
                citation,
                title,
                body.strip(),
                confidence,
                evidence,
                region,
            )
        )
        for branch_marker, branch_citation, branch_body in branches:
            result.append(
                Candidate(
                    record,
                    "ITEM",
                    branch_marker,
                    branch_citation,
                    None,
                    branch_body,
                    0.85,
                    ("LEXICAL_PREFIX", "INLINE_BRANCHED_ITEM"),
                    region,
                )
            )
    return result
