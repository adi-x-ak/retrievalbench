import json
from argparse import Namespace

import pytest

from retrievalbench.data import development_split
from retrievalbench.experiment import test as evaluate_test
from retrievalbench.experiment import tune


def test_existing_protocol_cannot_be_overwritten(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text('{"frozen": true}')
    with pytest.raises(ValueError, match="already frozen"):
        tune(Namespace(protocol=protocol))


def test_changed_protocol_cannot_overwrite_existing_evaluation(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text('{"changed": true}')
    results = tmp_path / "results"
    results.mkdir()
    (results / "evaluation.json").write_text(json.dumps({"protocol_sha256": "different"}))
    with pytest.raises(ValueError, match="another frozen protocol"):
        evaluate_test(Namespace(protocol=protocol, results=results))


def test_exact_heldout_text_duplicates_are_removed_only_from_development():
    queries = {"train-1": "same", "train-2": "different", "test-1": "same"}
    labels = {"train-1": {"a": 1}, "train-2": {"b": 1}}
    clean, excluded = development_split(queries, labels)
    assert clean == {"train-2": {"b": 1}}
    assert excluded == ["train-1"]
    assert "test-1" in queries
