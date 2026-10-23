"""Run a retriever and a generator over a dataset and score both halves."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .answer import answer_relevance, context_utilisation, faithfulness
from .dataset import Dataset, Example
from .retrieval import (
    average,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# A retriever takes a question and returns an ordered list of passage ids.
Retriever = Callable[[str], list[str]]
# A generator takes (question, contexts) and returns an answer string.
Generator = Callable[[str, list[str]], str]


@dataclass
class ExampleResult:
    example_id: str
    retrieved: list[str]
    answer: str
    metrics: dict[str, float] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "retrieved": self.retrieved,
            "answer": self.answer,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
        }


@dataclass
class Report:
    dataset_name: str
    k: int
    retrieval: dict[str, float] = field(default_factory=dict)
    answer: dict[str, float] = field(default_factory=dict)
    results: list[ExampleResult] = field(default_factory=list)

    def public(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "k": self.k,
            "retrieval": {k: round(v, 4) for k, v in self.retrieval.items()},
            "answer": {k: round(v, 4) for k, v in self.answer.items()},
            "results": [r.public() for r in self.results],
        }


def evaluate_example(
    example: Example,
    retriever: Retriever,
    generator: Generator,
    *,
    k: int,
) -> ExampleResult:
    retrieved = list(retriever(example.question))[:k]
    lookup = {p.id: p.text for p in example.passages}
    contexts = [lookup[doc_id] for doc_id in retrieved if doc_id in lookup]
    answer = generator(example.question, contexts)
    metrics = {
        "recall@k": recall_at_k(retrieved, example.relevant_ids, k),
        "precision@k": precision_at_k(retrieved, example.relevant_ids, k),
        "mrr": reciprocal_rank(retrieved, example.relevant_ids),
        "ndcg@k": ndcg_at_k(retrieved, example.relevant_ids, k),
        "faithfulness": faithfulness(answer, contexts),
        "answer_relevance": answer_relevance(answer, example.question),
        "context_utilisation": context_utilisation(answer, contexts),
    }
    return ExampleResult(
        example_id=example.id, retrieved=retrieved, answer=answer, metrics=metrics
    )


def evaluate(
    dataset: Dataset,
    retriever: Retriever,
    generator: Generator,
    *,
    k: int = 5,
) -> Report:
    results = [evaluate_example(e, retriever, generator, k=k) for e in dataset.examples]
    retrieval_keys = ("recall@k", "precision@k", "mrr", "ndcg@k")
    answer_keys = ("faithfulness", "answer_relevance", "context_utilisation")
    return Report(
        dataset_name=dataset.name,
        k=k,
        retrieval={key: average([r.metrics[key] for r in results]) for key in retrieval_keys},
        answer={key: average([r.metrics[key] for r in results]) for key in answer_keys},
        results=results,
    )
