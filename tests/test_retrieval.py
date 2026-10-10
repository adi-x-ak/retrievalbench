from dataclasses import asdict

import numpy as np
import pytest

from retrievalbench.fusion import Router, positions, rankings, rrf
from retrievalbench.lexical import BM25


def test_bm25_matches_hand_computed_term_frequency_and_length_normalization():
    index = BM25(["alpha alpha beta", "alpha gamma", "beta"])
    idf = np.log(1 + 1.5 / 2.5)
    expected = np.array(
        [
            idf
            * (
                2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 3 / 2))
                + 2.2 / (1 + 1.2 * (0.25 + 0.75 * 3 / 2))
            ),
            idf,
            idf * 2.2 / (1 + 1.2 * (0.25 + 0.75 / 2)),
        ]
    )
    np.testing.assert_allclose(index.score("alpha beta"), expected, rtol=1e-12)
    np.testing.assert_array_equal(index.score("unknown"), np.zeros(3))


def test_rankings_have_stable_ties_exclude_identical_ids_and_drop_missing_sparse_terms():
    documents = ["a", "b", "c"]
    scores = np.array([[2.0, 2.0, 0.0]])
    ranks = positions(scores, documents, ["a"], sparse=True)
    assert rankings(ranks, documents, ["a"]) == {"a": ["b"]}
    with pytest.raises(ValueError, match="finite"):
        positions(np.array([[0, np.nan, 1]]), documents, ["q"])


def test_fusion_respects_missing_documents_and_query_specific_weights():
    sparse = np.array([[1, 2, 0], [1, 2, 0]])
    dense = np.array([[3, 2, 1], [3, 2, 1]])
    actual = rrf(sparse, dense, np.array([0.0, 1.0]), 60)
    np.testing.assert_allclose(actual[0], [1 / 61, 1 / 62, 0])
    np.testing.assert_allclose(actual[1], [1 / 63, 1 / 62, 1 / 61])


def test_router_learns_weight_selection_and_survives_json_shape_serialization():
    x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    utilities = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    router = Router.fit(x, utilities, [0.0, 1.0], penalty=1)
    np.testing.assert_array_equal(Router(**asdict(router)).predict(x), [0, 0, 1, 1])
