import json
from argparse import Namespace

import pytest

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
