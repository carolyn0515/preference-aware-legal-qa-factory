from pathlib import Path

from legal_qa_factory.silver.semantics.enricher import build_propositions

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    count = build_propositions(ROOT / "data" / "silver")
    print(f"[RESULT] PASSED: {count} proposition datasets published")
