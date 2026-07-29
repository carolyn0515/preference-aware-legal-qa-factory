from legal_qa_factory.common.hashing import sha256_text


def legal_node_id(source_id: str, version: str, path: str) -> str:
    return "SLV-" + sha256_text(f"{source_id}|{version}|{path}")


def proposition_id(
    legal_node_id_value: str,
    sequence: int,
    text: str,
) -> str:
    identity = f"{legal_node_id_value}|{sequence}|{sha256_text(text)}"
    return "PRP-" + sha256_text(identity)
