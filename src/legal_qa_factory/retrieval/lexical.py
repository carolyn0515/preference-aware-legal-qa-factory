from __future__ import annotations

import math
from collections import Counter
from typing import Any

from legal_qa_factory.retrieval.query_analysis import lexical_terms


class BM25Index:
    def __init__(
        self,
        documents: list[dict[str, Any]],
        *,
        text_field: str = "text",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.term_frequencies = [
            Counter(lexical_terms(document[text_field])) for document in documents
        ]
        self.lengths = [sum(values.values()) for values in self.term_frequencies]
        self.average_length = (
            sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        )
        document_frequency = Counter(
            term for frequencies in self.term_frequencies for term in frequencies
        )
        count = len(documents)
        self.idf = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        terms = lexical_terms(query)
        if not terms or not self.documents:
            return []
        results = []
        for index, frequencies in enumerate(self.term_frequencies):
            score = 0.0
            for term in terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1
                    - self.b
                    + self.b * self.lengths[index] / max(self.average_length, 1)
                )
                score += self.idf.get(term, 0.0) * (
                    frequency * (self.k1 + 1) / denominator
                )
            if score:
                matched = set(terms) & set(frequencies)
                results.append(
                    {
                        "document": self.documents[index],
                        "bm25_raw": score,
                        "query_coverage": len(matched) / len(set(terms)),
                        "matched_terms": sorted(matched),
                    }
                )
        results.sort(
            key=lambda row: (
                -row["bm25_raw"],
                row["document"]["proposition_id"],
            )
        )
        if results:
            maximum = results[0]["bm25_raw"]
            for result in results:
                result["bm25_normalized"] = result["bm25_raw"] / maximum
        return results[:limit]
