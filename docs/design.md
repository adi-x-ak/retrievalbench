# Protocol and limits

## Inputs and baselines

The source is the [official BEIR SciFact archive](https://github.com/beir-cellar/beir#beers-available-datasets).
The archive SHA-256 is pinned in `data.py`; decompressed file hashes are stored
in the frozen protocol. Titles and abstracts are joined with one space. Document
IDs are sorted lexicographically before indexing. Train labels provide 809
development queries; the official test split is reserved for final evaluation.
Two development IDs (`1291`, `871`) have exact text matches among reserved query
texts. They are excluded from calibration, leaving 807 queries. All 300 official
test queries remain. Exclusions use text/ID membership without held-out relevance
labels and are recorded in the protocol. The initial protocol was aborted by the
overlap guard before scoring; its unmodified contents remain under `protocols/`.
The pipeline rejects exact query-ID/text overlap between these partitions.

BM25 uses lowercase Unicode word runs, an explicit 33-word stop list, Snowball
English stemming, and the positive Robertson IDF form
`log(1 + (N - df + 0.5)/(df + 0.5))`. Length counts analyzed tokens; repeated query
terms contribute multiplicatively. The development grid is k1 in {0.6,1.2,1.6}
and b in {0,0.5,0.75}. This is the implementation in this repository, not a claim
to reproduce Anserini's analyzer or its published score.

The compact dense baseline is
[BGE-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), MIT licensed,
revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. It uses normalized CLS vectors,
dimension 384, float32, maximum length 512, batch size 16, and exact cosine search.
Both plain queries and the model-card retrieval instruction are evaluated on
development labels; the better mode is frozen. Documents receive no instruction.
The original model's pretraining data overlap is not independently audited.

## Fusion and calibration

For each query, retain up to 1,000 ranks from each retriever. Sparse documents with
zero scores are absent. The fusion score is
`(1-w)/(k+sparse_rank) + w/(k+dense_rank)`; missing terms contribute zero.
Fixed fusion tunes k in {10,30,60} and w in {0,0.1,...,1} on development nDCG@10.
Including endpoints lets either standalone retriever win rather than forcing
a weaker fusion configuration.

The experimental router predicts the nDCG utility of alternative weights from
seven query/retrieval features: log word count, maximum sparse score, relative
sparse score gap, maximum dense score, dense gap, and rank overlap at 10 and 100.
Its transparent ridge coefficients and feature scales are serialized as JSON.
No opaque model service or pickle artifact is required.

A seeded 80/20 development partition fits and selects ridge regularization in
{1,10,100,1000}. The comparison is the fixed fusion selected on the full development
set; this grants the fixed baseline access to all development labels. Routing is
selected only if it improves internal held-out nDCG. A selected router is then
refitted on all development queries. Both method selection and hyperparameters
are in-domain calibration; internal selection scores are not an unbiased final
performance estimate. The official test split provides the final comparison.

`protocol.json` is frozen in Git before final evaluation. It records all baseline
grids, the selection outcome, input/embedding hashes, and deployed settings.
The CLI refuses to overwrite a protocol or reuse an evaluation directory with a
different protocol hash. These are workflow safeguards, not tamper-proof controls.
The first study uses its exact calibration cache. `--reproduction` permits an
independently encoded cache with the same model/input revisions and records both
artifact hashes; cache-byte identity is not promised across runtimes or ZIP
timestamps. A reproduction does not retune the frozen method.

## Evaluation

Metrics use the `pytrec_eval` interface from `pytrec-eval-terrier`, backed by
TREC's evaluator, matching BEIR's nDCG/recall definitions. Graded gains are linear.
Every labeled query must be present; an empty ranking scores zero. Rankings use
lexicographic document-ID ties and exclude identical query/document IDs, matching
BEIR's exclusion convention. Exported strictly decreasing scores preserve that
declared ordering through the evaluator. Report nDCG@10 and recall@100.

Paired bootstrap resamples complete queries 10,000 times with seed 42. Report
candidate-minus-baseline mean differences and percentile 95% intervals. An
interval crossing zero is inconclusive evidence of improvement. Compare against
tuned BM25, development-selected BGE-small, and tuned fixed weighted RRF; do not
omit a stronger baseline when it wins.

The pipeline saves every per-query metric and compressed ranking list. Corpus,
model files, and embeddings are excluded from Git. No corpus abstracts are
republished in the result artifacts. Dataset terms belong to the original
[SciFact project](https://github.com/allenai/scifact); this repository's MIT license
applies to its own implementation.

## Resource accounting

All methods share the same embedding model and exact dense index. Fusion adds a
lexical index and query scoring; routing adds seven features and a small matrix
multiply. There is no cross-encoder, generation model, or model fine-tuning.
Report corpus encoding separately from query encoding and cached vector search.
Preparing both query modes is development overhead, not per-query deployed cost.
Stored dense-index bytes exclude model weights and temporary tensors.

CPU timing on one laptop is not a production latency benchmark. Search times are
batch totals and exclude query embedding, model downloads, metric computation,
and HTTP transport. No GPU/cluster or state-of-the-art performance claim follows
from this study. The initial cross-encoder budget-allocation idea remains a
future experiment, not an implemented feature.

## References

- [BEIR](https://arxiv.org/abs/2104.08663): benchmark and evaluation framework.
- [BEIR evaluator](https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py): metric and identical-ID conventions.
- [Reciprocal rank fusion](https://doi.org/10.1145/1571941.1572114): original method.
- [BGE model card](https://huggingface.co/BAAI/bge-small-en-v1.5): encoder pooling and query instruction.
