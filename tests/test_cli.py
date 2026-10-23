import json
from pathlib import Path

from raglens.cli import main, render
from raglens.dataset import load_dataset
from raglens.evaluate import evaluate
from raglens.cli import _extractive_generator, _lexical_retriever

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_builtin_pipeline_retrieves_the_right_passage():
    dataset = load_dataset(EXAMPLES / "dataset.json")
    example = dataset.examples[0]
    retrieved = _lexical_retriever(example.passages)(example.question)
    assert retrieved[0] == "p1"


def test_builtin_pipeline_is_faithful():
    dataset = load_dataset(EXAMPLES / "dataset.json")
    report = evaluate(
        dataset,
        lambda q: _lexical_retriever(dataset.examples[0].passages)(q),
        _extractive_generator,
        k=3,
    )
    assert report.answer["faithfulness"] == 1.0


def test_render_shows_both_halves():
    dataset = load_dataset(EXAMPLES / "dataset.json")
    report = evaluate(dataset, lambda q: ["p1"], _extractive_generator, k=1)
    text = render(report)
    assert "Retrieval" in text
    assert "Answer" in text


def test_cli_runs_the_example(tmp_path):
    out = tmp_path / "r.json"
    code = main([str(EXAMPLES / "dataset.json"), "--json", str(out)])
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["retrieval"]["recall@k"] == 1.0


def test_cli_missing_dataset_returns_2():
    assert main(["/nonexistent/dataset.json"]) == 2
