# Research and code standards

Keep corpus analysis, embeddings, fusion, metrics, and experiment phases explicit.
Public reports must name the exact dataset, model revision, split, input hashes,
and preprocessing. Do not call in-domain calibration zero-shot evaluation.

Tune only on development labels. Freeze `protocol.json` in Git before running
the official test evaluation. Never change the published protocol based on test
scores. Reproduction may rerun the same protocol; new hypotheses need a new
protocol and an explicitly declared evaluation history.

Report all baselines, per-query results, and uncertainty. A point improvement
with an interval crossing zero is inconclusive. A SciFact result is not an
overall BEIR record or SOTA claim. Do not hide failed candidates.

Run `ruff check .`, `ruff format --check .`, and `pytest -q` before commits.
