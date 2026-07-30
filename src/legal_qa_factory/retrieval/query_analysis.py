from __future__ import annotations

import re
from typing import Any

TOKEN = re.compile(r"[가-힣A-Za-z0-9]+")
ARTICLE = re.compile(r"제(?P<number>\d+)조(?:의(?P<branch>\d+))?")
STOPWORDS = frozenset(
    {
        "그",
        "및",
        "또는",
        "관한",
        "경우",
        "대한",
        "있습니다",
        "합니다",
        "해야",
        "수",
        "등",
        "이",
        "가",
        "은",
        "는",
        "을",
        "를",
    }
)
ROLE_MARKERS = {
    "EXCEPTION_NOTICE": ("다만", "예외", "제외", "불구하고"),
    "CONDITION": ("경우", "요건", "해당", "때에는"),
    "PRACTICAL_GUIDANCE": ("확인", "검토", "사실관계", "계약서"),
    "PROCEDURE": ("신청", "통지", "제출", "절차", "먼저"),
    "SANCTION_NOTICE": ("과징금", "벌금", "과태료", "손해배상", "제재"),
}


def lexical_terms(text: str) -> list[str]:
    words = [
        value.casefold()
        for value in TOKEN.findall(text)
        if len(value) > 1 and value.casefold() not in STOPWORDS
    ]
    compact = re.sub(r"\s+", "", text)
    bigrams = [
        f"§{compact[index:index + 2]}"
        for index in range(len(compact) - 1)
        if all("가" <= char <= "힣" for char in compact[index : index + 2])
    ]
    return list(dict.fromkeys([*words, *bigrams]))


def analyze_claim(text: str) -> dict[str, Any]:
    roles = [
        role
        for role, markers in ROLE_MARKERS.items()
        if any(marker in text for marker in markers)
    ]
    if not roles:
        roles = ["CONCLUSION"]
    citations = [
        f"제{match.group('number')}조"
        + (f"의{match.group('branch')}" if match.group("branch") else "")
        for match in ARTICLE.finditer(text)
    ]
    return {
        "keywords": lexical_terms(text),
        "answer_roles": roles,
        "explicit_citations": citations,
        "has_negation": any(
            marker in text for marker in ("아니", "금지", "못한다", "없다")
        ),
    }
