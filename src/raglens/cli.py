"""Command line entry point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dataset import DatasetError, load_dataset
from .evaluate import evaluate
from .text import token_set


def _lexical_retriever(example_passages):
    """A dependency-free retriever: rank passages by token overlap with the question."""

    def retrieve(question: str) -> list[str]:
        question_tokens = token_set(question)
        scored = []
        for passage in example_passages:
            overlap = len(question_tokens & token_set(passage.text))
            scored.append((overlap, passage.id))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [doc_id for _, doc_id in scored]

    return retrieve


def _extractive_generator(question: str, contexts: list[str]) -> str:
    """A dependency-free generator: return the most relevant context sentence."""
    from .text import split_sentences

    question_tokens = token_set(question)
    best, best_score = "", -1.0
    for context in contexts:
        for sentence in split_sentences(context):
            score = len(question_tokens & token_set(sentence))
            if score > best_score:
                best, best_score = sentence, score
    return best


def render(report) -> str:
    lines = [
        f"Dataset: {report.dataset_name}   (k={report.k})",
        "",
        "Retrieval",
    ]
    for key, value in report.retrieval.items():
        lines.append(f"  {key:<18} {value:.3f}")
    lines.append("")
    lines.append("Answer")
    for key, value in report.answer.items():
        lines.append(f"  {key:<18} {value:.3f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-lens", description="Evaluate a RAG pipeline")
    parser.add_argument("dataset", help="path to the evaluation dataset JSON")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--json", metavar="PATH")
    parser.add_argument("--min-faithfulness", type=float, default=0.0)
    args = parser.parse_args(argv)

    try:
        dataset = load_dataset(args.dataset)
    except (DatasetError, OSError) as exc:
        print(f"could not load dataset: {exc}", file=sys.stderr)
        return 2

    # The built-in retriever/generator are per-example, so evaluate manually.
    from .evaluate import Report, evaluate_example

    results = [
        evaluate_example(
            example,
            _lexical_retriever(example.passages),
            _extractive_generator,
            k=args.k,
        )
        for example in dataset.examples
    ]
    from .retrieval import average

    report = Report(
        dataset_name=dataset.name,
        k=args.k,
        retrieval={
            key: average([r.metrics[key] for r in results])
            for key in ("recall@k", "precision@k", "mrr", "ndcg@k")
        },
        answer={
            key: average([r.metrics[key] for r in results])
            for key in ("faithfulness", "answer_relevance", "context_utilisation")
        },
        results=results,
    )
    print(render(report))

    if args.json:
        Path(args.json).write_text(json.dumps(report.public(), indent=2), encoding="utf-8")
        print(f"\nJSON report written to {args.json}")

    return 0 if report.answer["faithfulness"] >= args.min_faithfulness else 1


if __name__ == "__main__":
    sys.exit(main())
