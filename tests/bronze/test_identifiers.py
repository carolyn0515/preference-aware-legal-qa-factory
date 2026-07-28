from legal_qa_factory.bronze.identifiers import build_bronze_record_id


def test_record_id_is_deterministic() -> None:
    arguments = ("RAW-abc", 1, 0, "text-hash")
    assert build_bronze_record_id(*arguments) == build_bronze_record_id(*arguments)


def test_record_id_changes_with_location() -> None:
    first = build_bronze_record_id("RAW-abc", 1, 0, "text-hash")
    second = build_bronze_record_id("RAW-abc", 2, 0, "text-hash")
    assert first != second
