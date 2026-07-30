from legal_qa_factory.generation.expansion import expand_rows


def seed(question: str = "대금은 언제 지급하나요?") -> dict:
    return {
        "question": question,
        "answer": "법정 기한까지 지급해야 합니다.",
        "customer_id": "CUSTOMER",
        "reference_version": "seed_v1",
        "source_kind": "SYNTHETIC",
        "observed_evidence_ids": ["PRP-1"],
        "metadata": {"topic": "payment"},
    }


def plan() -> dict:
    return {
        "dataset": {
            "customer_id": "CUSTOMER",
            "reference_version": "expanded_v1",
            "source_kind": "SYNTHETIC",
            "target_count": 2,
        },
        "seed": {"expected_count": 1},
        "generation": {
            "include_seed_form": True,
            "variants_per_seed": 1,
            "method": "test",
            "variants": [
                {
                    "id": "scenario",
                    "question_type": "SCENARIO",
                    "difficulty": "INTERMEDIATE",
                    "prefix": "계약 검토 중입니다.",
                }
            ],
        },
        "quality": {
            "require_evidence": True,
            "require_parent_example_id": True,
            "reject_duplicate_questions": True,
            "allowed_question_types": ["BASE", "SCENARIO"],
            "allowed_difficulties": ["BASIC", "INTERMEDIATE"],
        },
    }


def test_expansion_preserves_evidence_and_groups_variants() -> None:
    result = expand_rows([seed()], plan())

    assert len(result.rows) == 2
    assert result.rows[0]["observed_evidence_ids"] == ["PRP-1"]
    assert (
        result.rows[0]["metadata"]["parent_example_id"]
        == result.rows[1]["metadata"]["parent_example_id"]
    )
    assert result.manifest["split_policy"] == "GROUP_BY_PARENT_EXAMPLE_ID"


def test_expansion_is_deterministic() -> None:
    first = expand_rows([seed()], plan())
    second = expand_rows([seed()], plan())

    assert first.rows == second.rows
    assert first.manifest == second.manifest
