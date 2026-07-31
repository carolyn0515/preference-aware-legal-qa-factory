from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from legal_qa_factory.common.hashing import sha256_text
from legal_qa_factory.retrieval.query_analysis import lexical_terms

QUESTION_MARKERS = {
    "CONDITION_LOOKUP": ("경우", "조건", "요건", "사유", "해당"),
    "EXCEPTION_LOOKUP": ("예외", "제외", "다만"),
    "PROCEDURE": ("어떻게", "절차", "방법", "신청", "요청"),
    "DEADLINE": ("언제", "기한", "기간", "며칠", "일 이내"),
    "SANCTION": ("처벌", "제재", "과징금", "벌금", "과태료"),
    "DEFINITION": ("무엇", "뜻", "정의"),
}

def question_features(question: str) -> dict[str, Any]:
    intents = [
        intent
        for intent, markers in QUESTION_MARKERS.items()
        if any(marker in question for marker in markers)
    ]
    if not intents:
        intents = ["GENERAL_LEGAL_QA"]
    return {
        "question_terms": lexical_terms(question),
        "question_intents": intents,
        "has_explicit_citation": bool(
            re.search(r"제\d+조(?:의\d+)?", question)
        ),
        "asks_condition": "CONDITION_LOOKUP" in intents,
        "asks_exception": "EXCEPTION_LOOKUP" in intents,
        "asks_procedure": "PROCEDURE" in intents,
        "asks_deadline": "DEADLINE" in intents,
        "asks_sanction": "SANCTION" in intents,
    }


def abstract_retrieval_actions(traversal_flow: list[str]) -> list[str]:
    actions = []
    for step in traversal_flow:
        relation = step.split(":", 1)[0]
        action = {
            "ANCHOR": "SEARCH_ANCHOR",
            "CHILD_ENUMERATION": "EXPAND_CHILDREN",
            "PARENT_CONTEXT": "EXPAND_PARENT",
            "SAME_ARTICLE_ROLE": "SEARCH_ROLE_SIBLINGS",
            "REFERENCED_ARTICLE": "FOLLOW_ARTICLE_REFERENCE",
            "IMPLEMENTING_DECREE": "FOLLOW_DECREE_DELEGATION",
        }.get(relation)
        if action:
            actions.append(action)
    return list(dict.fromkeys(actions))


def pattern_identity(
    answer_flow: list[str], retrieval_actions: list[str]
) -> str:
    canonical = json.dumps(
        {
            "answer_flow": answer_flow,
            "retrieval_actions": retrieval_actions,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "RBP-" + sha256_text(canonical)[:16]


def pattern_family(
    reference_intent: str,
    answer_flow: list[str],
    retrieval_actions: list[str],
) -> str:
    roles = set(answer_flow)
    actions = set(retrieval_actions)
    if reference_intent == "sanction" or "SANCTION_NOTICE" in roles:
        return "SANCTION_REMEDY"
    if reference_intent == "deadline":
        return "DEADLINE_CALCULATION"
    if (
        reference_intent in {"procedure", "permission"}
        or "PROCEDURE" in roles
        or "FOLLOW_DECREE_DELEGATION" in actions
        or "FOLLOW_ARTICLE_REFERENCE" in actions
    ):
        return "PROCEDURE_DELEGATION"
    if (
        reference_intent in {"condition_lookup", "exception_lookup"}
        or "CONDITION" in roles
        or "EXCEPTION_NOTICE" in roles
    ):
        return "CONDITION_EXCEPTION"
    return "DIRECT_RULE"


def routing_family(reference_intent: str, lineage_family: str) -> str:
    if reference_intent == "sanction":
        return "SANCTION_REMEDY"
    if reference_intent == "deadline":
        return "DEADLINE_CALCULATION"
    if reference_intent in {"procedure", "permission"}:
        return "PROCEDURE_DELEGATION"
    if reference_intent in {"condition_lookup", "exception_lookup"}:
        return "CONDITION_EXCEPTION"
    if reference_intent in {
        "prohibition",
        "document_issuance",
    }:
        return "DIRECT_RULE"
    return lineage_family


def compile_training_rows(
    qa_rows: list[dict[str, Any]],
    qa_flows: list[dict[str, Any]],
    tree_flows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    flows_by_id = {row["reference_qa_id"]: row for row in qa_flows}
    trees_by_id = {row["reference_qa_id"]: row for row in tree_flows}
    training_rows = []
    for qa in qa_rows:
        qa_id = qa["reference_qa_id"]
        if qa_id not in flows_by_id or qa_id not in trees_by_id:
            raise ValueError(f"missing lineage flow for {qa_id}")
        answer_flow = flows_by_id[qa_id]["answer_flow"]
        retrieval_actions = abstract_retrieval_actions(
            trees_by_id[qa_id]["traversal_flow"]
        )
        pattern_id = pattern_identity(answer_flow, retrieval_actions)
        customer_gold = qa["source_kind"] == "CUSTOMER_GOLD"
        metadata = qa.get("metadata")
        if metadata is None:
            metadata = json.loads(qa.get("metadata_json", "{}"))
        parent_example_id = metadata.get(
            "parent_example_id", qa["reference_qa_id"]
        )
        reference_intent = metadata.get("intent", "unknown")
        reference_topic = metadata.get("topic", "unknown")
        generation_variant_id = metadata.get("variant_id", "unknown")
        ranking_training_eligible = (
            generation_variant_id != "exception_check"
        )
        lineage_family = pattern_family(
            reference_intent, answer_flow, retrieval_actions
        )
        training_rows.append(
            {
                "reference_qa_id": qa_id,
                "parent_example_id": parent_example_id,
                "customer_id": qa["customer_id"],
                "reference_version": qa["reference_version"],
                "source_kind": qa["source_kind"],
                "question": qa["question"],
                **question_features(qa["question"]),
                "pattern_id": pattern_id,
                "pattern_family": routing_family(
                    reference_intent, lineage_family
                ),
                "lineage_pattern_family": lineage_family,
                "family_label_source": (
                    "REFERENCE_INTENT_ROUTING_V2"
                ),
                "reference_intent": reference_intent,
                "reference_topic": reference_topic,
                "generation_variant_id": generation_variant_id,
                "ranking_training_eligible": ranking_training_eligible,
                "ranking_exclusion_reason": (
                    None
                    if ranking_training_eligible
                    else "QUESTION_INTENT_CHANGED_WITHOUT_ANSWER_REGENERATION"
                ),
                "answer_flow": answer_flow,
                "retrieval_actions": retrieval_actions,
                "requires_decree": (
                    "FOLLOW_DECREE_DELEGATION" in retrieval_actions
                ),
                "requires_exception_search": (
                    "EXCEPTION_NOTICE" in answer_flow
                ),
                "requires_child_expansion": (
                    "EXPAND_CHILDREN" in retrieval_actions
                ),
                "label_provenance": "INFERRED",
                "sample_weight": 0.5 if customer_gold else 0.2,
                "production_training_eligible": False,
            }
        )

    counts = Counter(row["pattern_id"] for row in training_rows)
    patterns = []
    for pattern_id in sorted(counts):
        example = next(
            row for row in training_rows if row["pattern_id"] == pattern_id
        )
        patterns.append(
            {
                "pattern_id": pattern_id,
                "answer_flow": example["answer_flow"],
                "retrieval_actions": example["retrieval_actions"],
                "support_count": counts[pattern_id],
                "support_rate": round(
                    counts[pattern_id] / len(training_rows), 4
                ),
                "status": (
                    "INSUFFICIENT_SAMPLE"
                    if len(training_rows) < 10 or counts[pattern_id] < 5
                    else "CANDIDATE"
                ),
            }
        )
    return training_rows, {
        "schema_version": "1.0",
        "example_count": len(training_rows),
        "patterns": patterns,
    }
