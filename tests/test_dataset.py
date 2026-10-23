import json

import pytest

from raglens.dataset import DatasetError, load_dataset, parse_dataset


def test_parse_a_minimal_dataset():
    dataset = parse_dataset({"name": "d", "examples": [{"id": "a", "question": "q"}]})
    assert dataset.name == "d"
    assert dataset.examples[0].question == "q"


def test_missing_examples_is_an_error():
    with pytest.raises(DatasetError):
        parse_dataset({"name": "d"})


def test_empty_examples_is_an_error():
    with pytest.raises(DatasetError):
        parse_dataset({"examples": []})


def test_example_without_question_is_rejected():
    with pytest.raises(DatasetError):
        parse_dataset({"examples": [{"id": "a"}]})


def test_malformed_passage_is_rejected():
    with pytest.raises(DatasetError):
        parse_dataset({"examples": [{"question": "q", "passages": [{"id": "p"}]}]})


def test_ids_are_generated_when_absent():
    dataset = parse_dataset({"examples": [{"question": "q"}]})
    assert dataset.examples[0].id == "ex_0"


def test_passage_lookup_flattens_every_example():
    dataset = parse_dataset(
        {
            "examples": [
                {"question": "q", "passages": [{"id": "p1", "text": "one"}]},
                {"question": "r", "passages": [{"id": "p2", "text": "two"}]},
            ]
        }
    )
    assert dataset.passage_lookup() == {"p1": "one", "p2": "two"}


def test_load_dataset_from_disk(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(json.dumps({"examples": [{"question": "q"}]}))
    assert load_dataset(path).examples[0].question == "q"


def test_load_dataset_rejects_bad_json(tmp_path):
    path = tmp_path / "d.json"
    path.write_text("{nope")
    with pytest.raises(DatasetError):
        load_dataset(path)


def test_example_dataset_loads():
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "examples" / "dataset.json"
    assert len(load_dataset(example).examples) == 3
