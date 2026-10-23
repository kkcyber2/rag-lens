# RAG Lens

Measures whether a retrieval-augmented system is actually retrieving the right things
and answering from them — the two failures that look identical from the outside.

A RAG system can fail in two independent ways: it retrieves the wrong passages, or it
retrieves the right ones and then ignores them. RAG Lens scores both, separately.

## Metrics

**Retrieval**

| Metric | Question it answers |
|---|---|
| `recall@k` | Was a relevant passage in the top k? |
| `precision@k` | How much of the top k was relevant? |
| `mrr` | How high did the first relevant passage rank? |
| `ndcg@k` | Was the *ordering* good, not just the membership? |

**Answer**

| Metric | Question it answers |
|---|---|
| `faithfulness` | Is every claim in the answer supported by the retrieved context? |
| `answer_relevance` | Does the answer address the question? |
| `context_utilisation` | How much of the retrieved context was actually used? |

## Why both halves

Optimising retrieval alone hides a generation problem; optimising generation alone
hides a retrieval problem. Reporting them side by side is what makes the diagnosis
possible.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
rag-lens examples/dataset.json
rag-lens examples/dataset.json --json report.json
```

## As a library

```python
from raglens.dataset import load_dataset
from raglens.evaluate import evaluate

dataset = load_dataset("examples/dataset.json")
report = evaluate(dataset, retriever=my_retriever, generator=my_generator)
print(report.retrieval["recall@k"], report.answer["faithfulness"])
```

## Tests

```bash
pytest -q
```

## License

MIT — see `LICENSE`.
