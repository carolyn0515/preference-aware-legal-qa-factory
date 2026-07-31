from legal_qa_factory.blueprints.family_audit import _matrix


def test_confusion_matrix_contains_zero_cells() -> None:
    result = _matrix([("A", "A"), ("A", "B"), ("B", "B")])

    assert result == {
        "A": {"A": 1, "B": 1},
        "B": {"A": 0, "B": 1},
    }
