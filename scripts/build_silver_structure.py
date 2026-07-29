from pathlib import Path

from legal_qa_factory.silver.pipeline import build_silver_structure

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    count = build_silver_structure(ROOT / "data" / "bronze", ROOT / "data" / "silver")
    print(f"[RESULT] PASSED: {count} Silver datasets published")
