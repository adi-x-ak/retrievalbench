# RetrievalBench

[![Check](https://github.com/adi-x-ak/retrievalbench/actions/workflows/check.yml/badge.svg)](https://github.com/adi-x-ak/retrievalbench/actions/workflows/check.yml)

An inspectable retrieval experiment on **BEIR SciFact**: reproduce strong compact
baselines, calibrate hybrid fusion using development labels, freeze the method,
then evaluate the official test split.

The implementation includes BM25, pinned BGE-small embeddings, weighted
reciprocal-rank fusion, an optional query-conditioned ridge router, TREC metrics,
and paired bootstrap intervals. No model API or paid GPU is required.

## Study

- 5,183 scientific documents; 807 calibration queries after excluding two exact held-out text duplicates.
- `BAAI/bge-small-en-v1.5`, pinned to a full model revision, CLS pooling, normalized embeddings.
- BM25 parameters and query instruction selected using development nDCG@10.
- Fixed fusion tuned over 33 weight/reciprocal-rank-offset combinations.
- Query routing selected against fixed fusion on an internal development holdout.
- Official test labels are consumed only by the separate `test` phase after protocol freezing.

This is **in-domain calibration**, not zero-shot performance. A SciFact result
does not establish an overall BEIR record. The first test evaluation has not yet
been performed. The initial protocol was stopped by the text-overlap check before
any test score was computed and is archived in `protocols/aborted-overlap.json`.

## Reproduce

Python 3.11 or newer. Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[models,dev]'
retrievalbench prepare
retrievalbench test --protocol protocol.json --results reproduction --reproduction
```

`prepare` downloads the pinned dataset and model, then writes embeddings under
`cache/`. CPU is the default; `--device mps` is optional on compatible Macs. The
cache includes both instructed and plain queries. Model downloads use public
files and execute no remote model code.

The published protocol supplies frozen settings. To independently reproduce
development selection, use a fresh protocol path:

```bash
retrievalbench tune --protocol reproduction-protocol.json
```

Do not tune a new method against the published test scores and describe the
same test set as untouched. Dataset bytes, embedding bytes, model revision, and
protocol hashes are checked before evaluation. `--reproduction` permits a newly
encoded cache while retaining the frozen settings and recording both cache hashes;
hardware/runtime rounding and archive metadata can change the cache bytes.

## Read the code

| File | Responsibility |
|---|---|
| [lexical.py](src/retrievalbench/lexical.py) | English analysis, inverted postings, BM25 |
| [dense.py](src/retrievalbench/dense.py) | Pinned encoder and CLS pooling |
| [fusion.py](src/retrievalbench/fusion.py) | Stable ranks, weighted RRF, ridge routing |
| [evaluation.py](src/retrievalbench/evaluation.py) | TREC metrics and paired uncertainty |
| [experiment.py](src/retrievalbench/experiment.py) | Prepare → tune → freeze → test |
| [data.py](src/retrievalbench/data.py) | Corpus provenance and split fingerprints |

```bash
ruff check .
ruff format --check .
pytest -q
```

The unit suite uses small numerical cases and downloads no model. Full evaluation
uses all official SciFact test queries and saves every per-query score and ranking.
MIT-licensed implementation; dataset/model provenance and the complete protocol
are documented in [design.md](docs/design.md).
