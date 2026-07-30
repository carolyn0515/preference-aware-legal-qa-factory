from __future__ import annotations

import argparse
from pathlib import Path

from legal_qa_factory.evaluation.legal_function_audit import build_audit

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a token-free legal-function audit and review sample."
    )
    parser.add_argument("--sample-size", type=int, default=80)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "artifacts" / "legal_function_audits",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_audit(
        silver_dir=ROOT / "data" / "silver",
        output_dir=args.output_dir,
        sample_size=args.sample_size,
    )
    print(f"[AUDIT] status={report['status']}")
    print(
        "[COUNTS] "
        f"corpus={report['corpus_count']} "
        f"classified={report['classified_count']} "
        f"classification_sample={report['classification_sample_count']} "
        f"review_sample={report['review_sample_count']} "
        f"suspicious={report['suspicious_count']}"
    )
    if not report["evaluation_allowed"]:
        print(f"[BLOCKED] {report['blocking_reason']}")
    print(f"[OUTPUT] {args.output_dir}")


if __name__ == "__main__":
    main()
