from legal_qa_factory.silver.semantics.propositions import split_propositions


def node(text: str) -> dict:
    return {
        "legal_node_id": "NODE-1",
        "article_node_id": "ARTICLE-1",
        "source_id": "SOURCE-1",
        "source_version_hash": "VERSION-1",
        "text": text,
        "bronze_record_ids": ["BRZ-1"],
    }


def test_splits_korean_declarative_sentences() -> None:
    result = split_propositions(
        node("원사업자는 서면을 발급하여야 한다. 다만, 예외로 할 수 있다.")
    )
    assert [item.text for item in result] == [
        "원사업자는 서면을 발급하여야 한다.",
        "다만, 예외로 할 수 있다.",
    ]


def test_removes_amendment_note_from_propositions() -> None:
    result = split_propositions(
        node("이 법에 따른 수급사업자를 말한다.<개정 2024. 1. 1.>")
    )
    assert [item.text for item in result] == ["이 법에 따른 수급사업자를 말한다."]


def test_offsets_trace_to_original_text() -> None:
    original = "첫 번째로 한다. 두 번째로 한다."
    result = split_propositions(node(original))
    assert all(
        original[item.char_start : item.char_end] == item.text for item in result
    )


def test_does_not_split_inside_parentheses() -> None:
    result = split_propositions(
        node("중소기업자(협동조합을 포함한다. 이하 같다)가 아닌 사업자를 말한다.")
    )
    assert [item.text for item in result] == [
        "중소기업자(협동조합을 포함한다. 이하 같다)가 아닌 사업자를 말한다."
    ]
