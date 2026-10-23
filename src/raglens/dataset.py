"""The evaluation dataset."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class DatasetError(ValueError):
    """The dataset file is malformed."""


@dataclass
class Passage:
    id: str
    text: str


@dataclass
class Example:
    id: str
    question: str
    relevant_ids: set[str] = field(default_factory=set)
    passages: list[Passage] = field(default_factory=list)
    reference_answer: str | None = None


@dataclass
class Dataset:
    name: str
    examples: list[Example]

    def passage_lookup(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for example in self.examples:
            for passage in example.passages:
                out[passage.id] = passage.text
        return out


def parse_dataset(payload: dict[str, Any]) -> Dataset:
    if not isinstance(payload, dict):
        raise DatasetError("dataset must be a JSON object")
    raw_examples = payload.get("examples")
    if not isinstance(raw_examples, list) or not raw_examples:
        raise DatasetError("dataset must have a non-empty 'examples' list")
    examples: list[Example] = []
    for index, raw in enumerate(raw_examples):
        if not isinstance(raw, dict):
            raise DatasetError(f"example {index} is not an object")
        if "question" not in raw:
            raise DatasetError(f"example {index} has no 'question'")
        passages = []
        for passage in raw.get("passages", []):
            if not isinstance(passage, dict) or "id" not in passage or "text" not in passage:
                raise DatasetError(f"example {index} has a malformed passage")
            passages.append(Passage(id=str(passage["id"]), text=str(passage["text"])))
        examples.append(
            Example(
                id=str(raw.get("id") or f"ex_{index}"),
                question=str(raw["question"]),
                relevant_ids={str(r) for r in raw.get("relevant_ids", [])},
                passages=passages,
                reference_answer=raw.get("reference_answer"),
            )
        )
    return Dataset(name=str(payload.get("name", "unnamed")), examples=examples)


def load_dataset(path: str | Path) -> Dataset:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise DatasetError(f"dataset is not valid JSON: {exc}") from exc
    return parse_dataset(payload)
