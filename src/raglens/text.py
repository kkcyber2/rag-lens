"""Tokenisation and overlap helpers shared by the metrics."""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    this to was were will with""".split()
)

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    words = _WORD.findall(text.lower())
    if drop_stopwords:
        words = [w for w in words if w not in STOPWORDS]
    return [_stem(w) for w in words]


def _stem(word: str) -> str:
    """Very light suffix stripping so 'takes' matches 'take'.

    Without it, a question and an answer that differ only in inflection score as
    unrelated, which understates answer relevance.
    """
    for suffix in ("ing", "edly", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    # Normalise a trailing 'e' so 'take' and 'takes' land on the same stem.
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]
    return word


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def jaccard(a: str, b: str) -> float:
    left, right = token_set(a), token_set(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def split_sentences(text: str) -> list[str]:
    """Split on sentence punctuation, keeping non-empty trimmed pieces."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def coverage(claim: str, context: str, *, threshold: float = 0.5) -> float:
    """Fraction of the claim's content words that appear in the context."""
    claim_tokens = token_set(claim)
    if not claim_tokens:
        return 1.0
    context_tokens = token_set(context)
    return len(claim_tokens & context_tokens) / len(claim_tokens)
