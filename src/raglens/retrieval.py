"""Retrieval metrics."""
from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant id is in the top k, else 0.0."""
    if not relevant:
        return 1.0
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    return len(set(top) & relevant) / len(top)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1/rank of the first relevant result, or 0 if none is present."""
    for index, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / index
    return 0.0


def dcg(gains: list[float]) -> float:
    return sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalised discounted cumulative gain over the top k."""
    if not relevant:
        return 1.0
    gains = [1.0 if doc_id in relevant else 0.0 for doc_id in retrieved[:k]]
    ideal = [1.0] * min(len(relevant), k)
    best = dcg(ideal)
    if best == 0:
        return 0.0
    return dcg(gains) / best


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
