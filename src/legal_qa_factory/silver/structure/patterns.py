from __future__ import annotations

import re

ARTICLE = re.compile(
    r"^제(?P<number>\d+)조(?:의(?P<branch>\d+))?"
    r"(?:\((?P<title>[^)]+)\))?\s*(?P<body>.*)$"
)
PARAGRAPH = re.compile(r"^(?P<marker>[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])\s*(?P<body>.*)$")
ITEM = re.compile(r"^(?P<number>\d+)(?:의(?P<branch>\d+))?\.\s*(?P<body>.*)$")
INLINE_BRANCHED_ITEM = re.compile(r"\s+(?P<number>\d+)의(?P<branch>\d+)\.\s*")
SUBITEM = re.compile(r"^(?P<marker>[가-하])\.\s*(?P<body>.*)$")
ADDENDUM = re.compile(r"^(부칙|附則)(?:\s|$)")


def lexical_match(text: str) -> tuple[str, str, str, str | None, str] | None:
    match = ARTICLE.match(text)
    if match:
        number, branch = match.group("number"), match.group("branch")
        marker = number if branch is None else f"{number}-{branch}"
        citation = f"제{number}조" + (f"의{branch}" if branch else "")
        return "ARTICLE", marker, citation, match.group("title"), match.group("body")
    for pattern, kind, suffix in (
        (PARAGRAPH, "PARAGRAPH", ""),
        (ITEM, "ITEM", "호"),
        (SUBITEM, "SUBITEM", "목"),
    ):
        match = pattern.match(text)
        if match:
            if kind == "ITEM":
                number = match.group("number")
                branch = match.group("branch")
                marker = number if branch is None else f"{number}-{branch}"
                citation = f"{number}호" + (f"의{branch}" if branch else "")
            else:
                marker = match.group("marker")
                citation = marker + suffix
            body = match.group("body")
            if kind == "ITEM" and re.match(r"^\d{1,2}\.", body):
                return None
            return kind, marker, citation, None, body
    return None


def split_inline_branched_items(
    body: str,
) -> tuple[str, list[tuple[str, str, str]]]:
    matches = list(INLINE_BRANCHED_ITEM.finditer(body))
    if not matches:
        return body, []
    first_body = body[: matches[0].start()].strip()
    branches = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        number, branch = match.group("number"), match.group("branch")
        branches.append(
            (
                f"{number}-{branch}",
                f"{number}호의{branch}",
                body[match.end() : end].strip(),
            )
        )
    return first_body, branches
