from __future__ import annotations

from pathlib import Path

from legal_qa_factory.bronze.pipeline import build_bronze

PROJECT_ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    count = build_bronze(
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "bronze",
    )
    print(f"[RESULT] PASSED: {count} Bronze datasets published")
