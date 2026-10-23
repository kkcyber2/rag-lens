from raglens.answer import answer_relevance, context_utilisation, faithfulness


def test_faithful_answer_scores_one():
    contexts = ["Refunds can be requested within 30 days of delivery."]
    assert faithfulness("Refunds can be requested within 30 days of delivery.", contexts) == 1.0


def test_invented_sentence_lowers_faithfulness():
    contexts = ["Refunds can be requested within 30 days of delivery."]
    answer = "Refunds can be requested within 30 days of delivery. The CEO resigned yesterday."
    assert faithfulness(answer, contexts) == 0.5


def test_empty_answer_is_faithful():
    assert faithfulness("", ["anything"]) == 1.0


def test_answer_relevance_is_overlap():
    assert answer_relevance("shipping takes five days", "how long does shipping take") > 0.3


def test_unrelated_answer_has_low_relevance():
    assert answer_relevance("bananas are yellow", "how long does shipping take") < 0.1


def test_context_utilisation_counts_used_passages():
    contexts = ["Refunds within 30 days of delivery.", "Gift cards are non-refundable."]
    assert context_utilisation("Refunds within 30 days of delivery.", contexts) == 0.5


def test_context_utilisation_with_no_context_is_zero():
    assert context_utilisation("anything", []) == 0.0


def test_context_utilisation_with_empty_answer_is_zero():
    assert context_utilisation("", ["some context here"]) == 0.0
