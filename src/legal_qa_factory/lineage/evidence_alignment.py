from __future__ import annotations

import re
from typing import Any

from legal_qa_factory.retrieval.lexical import BM25Index
from legal_qa_factory.retrieval.query_analysis import analyze_claim, lexical_terms
from legal_qa_factory.retrieval.traversal import node_path

SELECTION_THRESHOLD = 0.55


def align_claim(
    *,
    claim: dict[str, Any],
    index: BM25Index,
    nodes_by_id: dict[str, dict[str, Any]],
    functions_by_proposition: dict[str, list[str]],
    legal_function_usable: bool,
    top_k: int = 5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    features = {
        "reference_claim_id": claim["reference_claim_id"],
        "reference_qa_id": claim["reference_qa_id"],
        "claim_sequence": claim["claim_sequence"],
        "text": claim["text"],
        **analyze_claim(claim["text"]),
    }
    candidates = []
    for result in index.search(claim["text"], limit=max(top_k * 4, 20)):
        proposition = result["document"]
        node = nodes_by_id[proposition["legal_node_id"]]
        article = nodes_by_id[node["article_node_id"]]
        query_words = set(features["keywords"])
        title_words = set(lexical_terms(article.get("title") or ""))
        compact_title = re.sub(r"\s+", "", article.get("title") or "").casefold()
        exact_compound_match = any(
            len(term) >= 4
            and not term.startswith("§")
            and term in compact_title
            for term in query_words
        )
        title_overlap = (
            len(query_words & title_words) / min(len(query_words), len(title_words))
            if query_words and title_words
            else 0.0
        )
        if exact_compound_match:
            title_overlap = 1.0
        citation_score = float(
            article["citation_label"] in features["explicit_citations"]
        )
        final_score = (
            0.45 * result["bm25_normalized"]
            + 0.25 * result["query_coverage"]
            + 0.20 * title_overlap
            + 0.10 * citation_score
        )
        candidates.append(
            {
                "reference_claim_id": claim["reference_claim_id"],
                "reference_qa_id": claim["reference_qa_id"],
                "claim_sequence": claim["claim_sequence"],
                "evidence_proposition_id": proposition["proposition_id"],
                "evidence_legal_node_id": node["legal_node_id"],
                "source_id": proposition["source_id"],
                "citation_label": node["citation_label"],
                "article_citation_label": article["citation_label"],
                "article_title": article.get("title"),
                "node_type": node["node_type"],
                "evidence_text": proposition["text"],
                "lineage_kind": "INFERRED",
                "retrieval_relation": "DIRECT_LEXICAL",
                "retrieval_path": node_path(node, nodes_by_id),
                "matched_terms": result["matched_terms"],
                "bm25_score": result["bm25_normalized"],
                "query_coverage": result["query_coverage"],
                "title_overlap": title_overlap,
                "citation_score": citation_score,
                "final_score": final_score,
                "legal_functions": (
                    functions_by_proposition.get(proposition["proposition_id"], [])
                    if legal_function_usable
                    else []
                ),
                "legal_function_usable": legal_function_usable,
            }
        )
    candidates.sort(
        key=lambda row: (-row["final_score"], row["evidence_proposition_id"])
    )
    for rank, candidate in enumerate(candidates[:top_k], start=1):
        candidate["rank"] = rank
        candidate["selected"] = (
            rank == 1 and candidate["final_score"] >= SELECTION_THRESHOLD
        )
        candidate["selection_status"] = (
            "CANDIDATE_EVIDENCE"
            if candidate["selected"]
            else "RETRIEVAL_CANDIDATE_ONLY"
        )
    return features, candidates[:top_k]
