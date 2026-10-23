"""Answer-quality metrics."""
from __future__ import annotations

from .text import coverage, jaccard, split_sentences, token_set


def faithfulness(answer: str, contexts: list[str]) -> float:
    """Fraction of the answer's sentences that are supported by the context.

    A sentence is supported when at least `threshold` of its content words appear
    somewhere in the retrieved context. This is a lexical proxy for entailment —
    cheap, deterministic, and good enough to catch an answer that invents facts.
    """
    sentences = split_sentences(answer)
    if not sentences:
        return 1.0
    joined = " ".join(contexts)
    supported = sum(1 for s in sentences if coverage(s, joined, threshold=0.5) >= 0.5)
    return supported / len(sentences)


def answer_relevance(answer: str, question: str) -> float:
    """How much of the question the answer actually addresses.

    Deliberately asymmetric: the question is the claim and the answer is the
    evidence, so this is the fraction of the question's content words that the
    answer covers. Symmetric Jaccard would penalise a short, complete answer for
    not repeating the question's filler words.
    """
    return coverage(question, answer)


def context_utilisation(answer: str, contexts: list[str]) -> float:
    """How much of the retrieved context the answer actually drew on."""
    if not contexts:
        return 0.0
    answer_tokens = token_set(answer)
    if not answer_tokens:
        return 0.0
    used = 0
    for context in contexts:
        context_tokens = token_set(context)
        if context_tokens and len(context_tokens & answer_tokens) / len(context_tokens) >= 0.2:
            used += 1
    return used / len(contexts)
