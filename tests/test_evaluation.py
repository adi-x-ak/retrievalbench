import math

import pytest

from retrievalbench.evaluation import evaluate, paired_interval


def test_trec_graded_gain_and_recall_are_correct():
    qrels = {"q": {"a": 2, "b": 1}}
    metrics, scores = evaluate(qrels, {"q": ["b", "a"]})
    expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert metrics["ndcg_cut_10"] == pytest.approx(expected)
    assert metrics["recall_100"] == scores["q"]["recall_100"] == 1


def test_empty_rankings_count_as_zero_and_missing_queries_are_rejected():
    metrics, _ = evaluate({"q": {"a": 1}}, {"q": []})
    assert metrics["ndcg_cut_10"] == 0
    with pytest.raises(ValueError, match="every labeled query"):
        evaluate({"q": {"a": 1}}, {})
    with pytest.raises(ValueError, match="duplicate"):
        evaluate({"q": {"a": 1}}, {"q": ["a", "a"]})


def test_paired_bootstrap_is_deterministic_and_zero_for_identical_runs():
    scores = {"q1": {"ndcg_cut_10": 1.0}, "q2": {"ndcg_cut_10": 0.5}}
    assert paired_interval(scores, scores, samples=1000)["ci95"] == [0, 0]
    better = {"q1": {"ndcg_cut_10": 1.0}, "q2": {"ndcg_cut_10": 1.0}}
    assert paired_interval(better, scores, samples=1000) == paired_interval(
        better, scores, samples=1000
    )
