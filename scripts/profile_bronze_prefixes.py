from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BRONZE_ROOT = PROJECT_ROOT / "data" / "bronze"

PREFIX_LENGTH = 30
EXAMPLE_LIMIT = 5

ARTICLE_PATTERN = re.compile(
    r"^제\d+조(?:의\d+)?"
)

CHAPTER_PATTERN = re.compile(
    r"^제\d+장"
)

SECTION_PATTERN = re.compile(
    r"^제\d+절"
)

PARAGRAPH_PATTERN = re.compile(
    r"^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]"
)

ITEM_PATTERN = re.compile(
    r"^\d+\.\s*"
)

SUBITEM_PATTERN = re.compile(
    r"^[가-하]\.\s*"
)

AMENDMENT_PATTERN = re.compile(
    r"^[<\[]?(?:전문개정|본조신설|개정|삭제)"
)

FAQ_SECTION_PATTERN = re.compile(
    r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+(?:\s|$)"
)

def classify_prefix(text: str) -> str:
    if ARTICLE_PATTERN.match(text):
        if re.match(r"^제\d+조의\d+", text):
            return "ARTICLE_BRANCH_CANDIDATE"
        return "ARTICLE_CANDIDATE"

    if CHAPTER_PATTERN.match(text):
        return "CHAPTER_CANDIDATE"

    if SECTION_PATTERN.match(text):
        return "SECTION_CANDIDATE"

    if PARAGRAPH_PATTERN.match(text):
        return "PARAGRAPH_CANDIDATE"

    if ITEM_PATTERN.match(text):
        return "NUMBERED_ITEM_CANDIDATE"

    if SUBITEM_PATTERN.match(text):
        return "KOREAN_SUBITEM_CANDIDATE"

    if AMENDMENT_PATTERN.match(text):
        return "AMENDMENT_CANDIDATE"

    if FAQ_SECTION_PATTERN.match(text):
        return "FAQ_SECTION_CANDIDATE"

    return "OTHER"

def load_bronze_records(
    parquet_path: Path,
) -> list[dict[str, Any]]:
    table = pq.read_table(
        parquet_path,
        columns=[
            "source_id",
            "source_type",
            "page_number",
            "block_index",
            "bronze_record_id",
            "normalized_text",
        ],
    )

    return table.to_pylist()

def profile_records(
    records: list[dict[str, Any]],
) -> None:
    signature_counts: Counter[str] = Counter()
    first_token_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        text = record["normalized_text"].strip()

        if not text:
            continue

        signature = classify_prefix(text)
        first_token = text.split(maxsplit=1)[0]

        signature_counts[signature] += 1
        first_token_counts[first_token] += 1

        if len(examples[signature]) < EXAMPLE_LIMIT:
            examples[signature].append(
                {
                    "page_number": record["page_number"],
                    "block_index": record["block_index"],
                    "prefix": text[:PREFIX_LENGTH],
                }
            )

    print("[STRUCTURAL SIGNATURES]")

    for signature, count in signature_counts.most_common():
        print()
        print(f"{signature}: {count}")

        for example in examples[signature]:
            print(
                f"  p{example['page_number']} "
                f"b{example['block_index']}: "
                f"{example['prefix']}"
            )

    print()
    print("[MOST COMMON FIRST TOKENS]")

    for token, count in first_token_counts.most_common(50):
        print(f"{count:>5}  {token}")

def main() -> int:
    parquet_paths = sorted(
        BRONZE_ROOT.glob("*/*/records.parquet")
    )

    if not parquet_paths:
        print("[ERROR] no Bronze Parquet files found")
        return 1

    for parquet_path in parquet_paths:
        records = load_bronze_records(parquet_path)

        if not records:
            continue

        source_id = records[0]["source_id"]

        print()
        print("=" * 80)
        print(f"[SOURCE] {source_id}")
        print(f"[ROWS] {len(records)}")
        print(f"[PATH] {parquet_path}")
        print("=" * 80)

        profile_records(records)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())