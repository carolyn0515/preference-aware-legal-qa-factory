from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def build_qa_flows(
    claim_features: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    features_by_qa: dict[str, list[dict[str, Any]]] = defaultdict(list)
    evidence_by_claim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for feature in claim_features:
        features_by_qa[feature["reference_qa_id"]].append(feature)
    for evidence in evidence_rows:
        if evidence["selected"]:
            evidence_by_claim[evidence["reference_claim_id"]].append(evidence)

    flows = []
    for qa_id, features in sorted(features_by_qa.items()):
        features.sort(key=lambda row: row["claim_sequence"])
        answer_flow = [
            role for feature in features for role in feature["answer_roles"]
        ]
        retrieval_flow = []
        citations = []
        grounded = 0
        for feature in features:
            selected = evidence_by_claim[feature["reference_claim_id"]]
            if selected:
                grounded += 1
                best = min(selected, key=lambda row: row["rank"])
                retrieval_flow.extend(best["retrieval_path"])
                citations.append(
                    f"{best['source_id']}:{best['article_citation_label']}"
                )
            else:
                retrieval_flow.append("NO_DIRECT_LEGAL_EVIDENCE")
        flows.append(
            {
                "reference_qa_id": qa_id,
                "answer_flow": answer_flow,
                "retrieval_flow": retrieval_flow,
                "evidence_citations": list(dict.fromkeys(citations)),
                "grounded_claim_count": grounded,
                "claim_count": len(features),
                "candidate_grounding_rate": grounded / len(features),
            }
        )
    return flows


def aggregate_flow_patterns(flows: list[dict[str, Any]]) -> dict[str, Any]:
    answer_counts = Counter(tuple(row["answer_flow"]) for row in flows)
    retrieval_counts = Counter(tuple(row["retrieval_flow"]) for row in flows)
    total = len(flows)

    def patterns(counts: Counter) -> list[dict[str, Any]]:
        values = []
        for flow, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        ):
            rate = count / total if total else 0.0
            values.append(
                {
                    "flow": list(flow),
                    "support_count": count,
                    "support_rate": round(rate, 4),
                    "status": (
                        "INSUFFICIENT_SAMPLE"
                        if total < 10
                        else (
                            "CANDIDATE"
                            if count >= 5 and rate >= 0.5
                            else "NOT_ESTABLISHED"
                        )
                    ),
                }
            )
        return values

    return {
        "schema_version": "1.0",
        "qa_count": total,
        "answer_flow_patterns": patterns(answer_counts),
        "retrieval_flow_patterns": patterns(retrieval_counts),
        "interpretation_gate": {
            "minimum_examples": 10,
            "minimum_support_count": 5,
            "minimum_support_rate": 0.5,
        },
    }


def build_tree_flows(
    qa_flows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    expansions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_by_qa: dict[str, list[dict[str, Any]]] = defaultdict(list)
    expansions_by_qa: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        if row["selected"]:
            selected_by_qa[row["reference_qa_id"]].append(row)
    for row in expansions:
        expansions_by_qa[row["reference_qa_id"]].append(row)

    rows = []
    for qa_flow in qa_flows:
        qa_id = qa_flow["reference_qa_id"]
        anchors = sorted(
            selected_by_qa[qa_id],
            key=lambda row: (row["claim_sequence"], row["rank"]),
        )
        contexts = sorted(
            expansions_by_qa[qa_id],
            key=lambda row: (
                row["reference_claim_id"],
                row["traversal_step"],
            ),
        )
        traversal = []
        articles = []
        for anchor in anchors:
            traversal.append(
                f"ANCHOR:{anchor['source_id']}:{anchor['article_citation_label']}"
            )
            articles.append(
                f"{anchor['source_id']}:{anchor['article_citation_label']}"
            )
            for context in contexts:
                if (
                    context["anchor_proposition_id"]
                    != anchor["evidence_proposition_id"]
                ):
                    continue
                traversal.append(
                    f"{context['expansion_relation']}:"
                    f"{context['source_id']}:"
                    f"{context['article_citation_label']}:"
                    f"{context['citation_label']}"
                )
                articles.append(
                    f"{context['source_id']}:"
                    f"{context['article_citation_label']}"
                )
        rows.append(
            {
                "reference_qa_id": qa_id,
                "traversal_flow": traversal,
                "visited_articles": list(dict.fromkeys(articles)),
                "anchor_count": len(anchors),
                "expanded_context_count": len(contexts),
            }
        )
    return rows
