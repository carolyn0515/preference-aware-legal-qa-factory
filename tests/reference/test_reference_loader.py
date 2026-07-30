import json

import pytest

from legal_qa_factory.reference.claims import split_answer_claims
from legal_qa_factory.reference.loader import load_jsonl


def write_jsonl(path, rows) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )


def valid_row(question: str = "대금은 언제 지급해야 하나요?") -> dict:
    return {
        "question": question,
        "answer": "원칙적으로 지급해야 합니다. 다만 예외가 있습니다.",
        "customer_id": "HYUNDAI_ENGINEERING",
        "reference_version": "synthetic_v1",
        "source_kind": "SYNTHETIC",
        "observed_evidence_ids": [],
        "metadata": {"intent": "deadline"},
    }


def test_loader_generates_stable_ids_and_preserves_input_hash(tmp_path) -> None:
    path = tmp_path / "gold.jsonl"
    write_jsonl(path, [valid_row()])
    first, first_hash = load_jsonl(path)
    second, second_hash = load_jsonl(path)
    assert first[0].reference_qa_id == second[0].reference_qa_id
    assert first_hash == second_hash
    assert first[0].source_row_number == 1


def test_loader_rejects_duplicate_normalized_questions(tmp_path) -> None:
    path = tmp_path / "gold.jsonl"
    write_jsonl(
        path,
        [valid_row("대금은 언제?"), valid_row("  대금은   언제?  ")],
    )
    with pytest.raises(ValueError, match="duplicate normalized questions"):
        load_jsonl(path)


def test_claim_split_preserves_exact_offsets() -> None:
    answer = "첫 번째 문장입니다. 두 번째 문장입니다."
    claims = split_answer_claims("RQA-test", answer)
    assert [item["text"] for item in claims] == [
        "첫 번째 문장입니다.",
        "두 번째 문장입니다.",
    ]
    for claim in claims:
        assert answer[claim["char_start"] : claim["char_end"]] == claim["text"]
