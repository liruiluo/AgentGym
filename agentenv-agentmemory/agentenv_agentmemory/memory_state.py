from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


@dataclass
class MemoryEntry:
    memory_id: str
    key: str
    value: str
    created_step: int
    updated_step: int
    access_count: int = 0

    def render(self) -> str:
        return f"[{self.memory_id}] {self.key}: {self.value}"


def tokenize_terms(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def tokenize(text: str) -> set[str]:
    return set(tokenize_terms(text))


def rank_memory_entries_bm25(
    query: str,
    entries: list[MemoryEntry],
    *,
    top_k: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[MemoryEntry, float]]:
    query_terms = tokenize(query)
    if not query_terms or not entries:
        return []

    documents = [tokenize_terms(f"{entry.key} {entry.value}") for entry in entries]
    doc_lengths = [len(document) for document in documents]
    avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
    if avg_doc_length <= 0.0:
        return []

    document_frequencies: Counter[str] = Counter()
    for document in documents:
        for term in set(document):
            if term in query_terms:
                document_frequencies[term] += 1

    total_docs = len(documents)
    scored: list[tuple[MemoryEntry, float]] = []
    for entry, document, doc_length in zip(entries, documents, doc_lengths):
        term_frequencies = Counter(document)
        score = 0.0
        for term in sorted(query_terms):
            term_frequency = term_frequencies.get(term, 0)
            if term_frequency <= 0:
                continue
            doc_frequency = document_frequencies.get(term, 0)
            idf = math.log(1.0 + (total_docs - doc_frequency + 0.5) / (doc_frequency + 0.5))
            denominator = term_frequency + k1 * (1.0 - b + b * doc_length / avg_doc_length)
            score += idf * ((term_frequency * (k1 + 1.0)) / denominator)
        if score > 0.0:
            scored.append((entry, score))

    scored.sort(key=lambda item: (-item[1], item[0].memory_id))
    return scored[: max(1, top_k)]
