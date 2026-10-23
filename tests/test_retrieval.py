from raglens.retrieval import (
    average,
    dcg,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_hits_when_relevant_is_in_top_k():
    assert recall_at_k(["a", "b", "c"], {"c"}, 3) == 1.0


def test_recall_misses_when_relevant_is_below_k():
    assert recall_at_k(["a", "b", "c"], {"c"}, 2) == 0.0


def test_recall_with_no_relevant_ids_is_one():
    assert recall_at_k(["a"], set(), 1) == 1.0


def test_precision_counts_the_relevant_share():
    assert precision_at_k(["a", "b", "c", "d"], {"a", "b"}, 4) == 0.5


def test_precision_of_empty_list_is_zero():
    assert precision_at_k([], {"a"}, 3) == 0.0


def test_reciprocal_rank_of_first_hit():
    assert reciprocal_rank(["a", "b"], {"a"}) == 1.0


def test_reciprocal_rank_of_second_hit():
    assert reciprocal_rank(["a", "b"], {"b"}) == 0.5


def test_reciprocal_rank_with_no_hit():
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_dcg_discounts_lower_ranks():
    assert dcg([1.0, 1.0]) > dcg([1.0, 0.0])


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == 1.0


def test_ndcg_penalises_a_bad_ordering():
    good = ndcg_at_k(["a", "b", "c"], {"a"}, 3)
    bad = ndcg_at_k(["c", "b", "a"], {"a"}, 3)
    assert good > bad


def test_ndcg_with_no_relevant_ids_is_one():
    assert ndcg_at_k(["a"], set(), 1) == 1.0


def test_average_of_empty_is_zero():
    assert average([]) == 0.0
