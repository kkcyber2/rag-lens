from raglens.dataset import parse_dataset
from raglens.evaluate import evaluate, evaluate_example


def make_dataset():
    return parse_dataset(
        {
            "name": "d",
            "examples": [
                {
                    "id": "a",
                    "question": "how long is the refund window",
                    "relevant_ids": ["p1"],
                    "passages": [
                        {"id": "p1", "text": "Refunds can be requested within 30 days of delivery."},
                        {"id": "p2", "text": "Shipping takes three to five business days."},
                    ],
                }
            ],
        }
    )


def perfect_retriever(question):
    return ["p1", "p2"]


def perfect_generator(question, contexts):
    return contexts[0] if contexts else ""


def test_perfect_pipeline_scores_well():
    report = evaluate(make_dataset(), perfect_retriever, perfect_generator, k=2)
    assert report.retrieval["recall@k"] == 1.0
    assert report.retrieval["mrr"] == 1.0
    assert report.answer["faithfulness"] == 1.0


def test_bad_retriever_lowers_retrieval_metrics():
    report = evaluate(make_dataset(), lambda q: ["p2", "p1"], perfect_generator, k=1)
    assert report.retrieval["recall@k"] == 0.0
    assert report.retrieval["mrr"] == 0.0


def test_hallucinating_generator_lowers_faithfulness():
    def liar(question, contexts):
        return "Refunds can be requested within 30 days of delivery. The moon is made of cheese."

    report = evaluate(make_dataset(), perfect_retriever, liar, k=2)
    assert report.answer["faithfulness"] < 1.0


def test_evaluate_example_returns_every_metric():
    result = evaluate_example(make_dataset().examples[0], perfect_retriever, perfect_generator, k=2)
    assert set(result.metrics) == {
        "recall@k", "precision@k", "mrr", "ndcg@k",
        "faithfulness", "answer_relevance", "context_utilisation",
    }


def test_retriever_returning_unknown_ids_is_handled():
    result = evaluate_example(
        make_dataset().examples[0], lambda q: ["nope"], perfect_generator, k=2
    )
    assert result.answer == ""


def test_public_dict_is_serialisable():
    import json

    json.dumps(evaluate(make_dataset(), perfect_retriever, perfect_generator).public())
