"""Prepare inputs, calibrate on train labels, freeze, then evaluate official test labels."""

import argparse
import gzip
import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from retrievalbench.data import fetch, fingerprint, input_hashes, load_qrels, load_text
from retrievalbench.dense import MODEL, REVISION, encode
from retrievalbench.evaluation import evaluate, paired_interval
from retrievalbench.fusion import Router, features, positions, rankings, rrf
from retrievalbench.lexical import BM25


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(args: argparse.Namespace) -> None:
    if not (args.data / "corpus.jsonl").exists():
        fetch(args.data)
    corpus, queries = load_text(args.data)
    documents, identifiers = sorted(corpus), sorted(queries)
    if (args.cache / "embeddings.npz").exists():
        raise ValueError("Embedding cache exists; use a fresh cache directory to prepare again")
    arrays, metadata = encode(
        [corpus[i] for i in documents], [queries[i] for i in identifiers], args.device
    )
    metadata.update(
        {"inputs": input_hashes(args.data), "documents": documents, "queries": identifiers}
    )
    args.cache.mkdir(parents=True, exist_ok=True)
    np.savez(args.cache / "embeddings.npz", **arrays)
    metadata["embeddings_sha256"] = fingerprint(args.cache / "embeddings.npz")
    write_json(args.cache / "metadata.json", metadata)


def inputs(
    args: argparse.Namespace,
) -> tuple[dict, dict[str, str], dict[str, str], dict[str, np.ndarray]]:
    metadata = json.loads((args.cache / "metadata.json").read_text())
    if metadata["inputs"] != input_hashes(args.data):
        raise ValueError("Dataset changed after encoding")
    if (metadata["model"], metadata["revision"]) != (MODEL, REVISION):
        raise ValueError("Embedding cache uses a different model revision")
    corpus, queries = load_text(args.data)
    if metadata["documents"] != sorted(corpus) or metadata["queries"] != sorted(queries):
        raise ValueError("Embedding IDs do not match the dataset")
    if metadata["embeddings_sha256"] != fingerprint(args.cache / "embeddings.npz"):
        raise ValueError("Embedding bytes changed after preparation")
    arrays = dict(np.load(args.cache / "embeddings.npz", allow_pickle=False))
    return metadata, corpus, queries, arrays


def dense_scores(arrays: dict, metadata: dict, qids: list[str], mode: str) -> np.ndarray:
    rows = {qid: row for row, qid in enumerate(metadata["queries"])}
    return arrays[mode][[rows[qid] for qid in qids]] @ arrays["corpus"].T


def fuse(
    sparse: np.ndarray, dense: np.ndarray, weights, k: float, docids: list[str], qids: list[str]
) -> dict:
    return rankings(
        positions(rrf(sparse, dense, weights, k), docids, qids, sparse=True), docids, qids
    )


def tune(args: argparse.Namespace) -> None:
    if args.protocol.exists():
        raise ValueError("Protocol is already frozen; use a different path for a new study")
    metadata, corpus, queries, arrays = inputs(args)
    docids = metadata["documents"]
    qrels = load_qrels(args.data, "train")
    qids = sorted(qrels)
    texts = [queries[qid] for qid in qids]
    lexical = BM25([corpus[docid] for docid in docids])
    bm25_grid = []
    best_sparse, best_sparse_scores = None, None
    for k1 in (0.6, 1.2, 1.6):
        for b in (0.0, 0.5, 0.75):
            scores = np.stack([lexical.score(text, k1, b) for text in texts])
            ranks = positions(scores, docids, qids, sparse=True)
            metrics, _ = evaluate(qrels, rankings(ranks, docids, qids))
            record = {"k1": k1, "b": b, **metrics}
            bm25_grid.append(record)
            if best_sparse is None or metrics["ndcg_cut_10"] > best_sparse["ndcg_cut_10"]:
                best_sparse, best_sparse_scores, sparse = record, scores, ranks
    dense_grid = []
    best_dense = None
    for mode in ("plain", "instructed"):
        scores = dense_scores(arrays, metadata, qids, mode)
        ranks = positions(scores, docids, qids)
        metrics, _ = evaluate(qrels, rankings(ranks, docids, qids))
        record = {"mode": mode, **metrics}
        dense_grid.append(record)
        if best_dense is None or metrics["ndcg_cut_10"] > best_dense["ndcg_cut_10"]:
            best_dense, best_dense_scores, dense = record, scores, ranks
    rrf_grid = []
    best_fusion = None
    for k in (10, 30, 60):
        for index in range(11):
            weight = index / 10
            metrics, _ = evaluate(qrels, fuse(sparse, dense, weight, k, docids, qids))
            record = {"k": k, "dense_weight": weight, **metrics}
            rrf_grid.append(record)
            if best_fusion is None or metrics["ndcg_cut_10"] > best_fusion["ndcg_cut_10"]:
                best_fusion = record
    x = features(sparse, dense, best_sparse_scores, best_dense_scores, texts)
    choices = sorted(set([0.0, 0.25, 0.5, 0.75, 1.0, best_fusion["dense_weight"]]))
    utilities = []
    for weight in choices:
        _, scores = evaluate(qrels, fuse(sparse, dense, weight, best_fusion["k"], docids, qids))
        utilities.append([scores[qid]["ndcg_cut_10"] for qid in qids])
    utilities = np.array(utilities).T
    indices = np.random.default_rng(42).permutation(len(qids))
    fitting, validation = indices[: int(0.8 * len(qids))], indices[int(0.8 * len(qids)) :]
    constant_score = float(utilities[validation, choices.index(best_fusion["dense_weight"])].mean())
    selection = [{"method": "constant", "ndcg_cut_10": constant_score}]
    best_score, best_penalty = constant_score, None
    for penalty in (1.0, 10.0, 100.0, 1000.0):
        router = Router.fit(x[fitting], utilities[fitting], choices, penalty)
        weights = router.predict(x[validation])
        columns = [choices.index(weight) for weight in weights]
        score = float(utilities[validation, columns].mean())
        selection.append({"method": "router", "penalty": penalty, "ndcg_cut_10": score})
        if score > best_score:
            best_score, best_penalty = score, penalty
    router = Router.fit(x, utilities, choices, best_penalty) if best_penalty is not None else None
    protocol = {
        "version": 1,
        "dataset": "BEIR SciFact",
        "inputs": metadata["inputs"],
        "model": MODEL,
        "revision": REVISION,
        "embeddings_sha256": metadata["embeddings_sha256"],
        "bm25": {"k1": best_sparse["k1"], "b": best_sparse["b"]},
        "dense_mode": best_dense["mode"],
        "rrf": {"k": best_fusion["k"], "dense_weight": best_fusion["dense_weight"]},
        "candidate": "router" if router else "rrf",
        "router": asdict(router) if router else None,
        "development": {
            "queries": len(qids),
            "bm25_grid": bm25_grid,
            "dense_grid": dense_grid,
            "rrf_grid": rrf_grid,
            "selection": selection,
            "router_fit_queries": len(fitting),
            "router_validation_queries": len(validation),
            "seed": 42,
        },
    }
    write_json(args.protocol, protocol)
    print(
        json.dumps(
            {
                "bm25": best_sparse,
                "dense": best_dense,
                "rrf": best_fusion,
                "selection": selection,
                "frozen": str(args.protocol),
            },
            indent=2,
        )
    )


def test(args: argparse.Namespace) -> None:
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    previous = args.results / "evaluation.json"
    if previous.exists() and json.loads(previous.read_text())["protocol_sha256"] != protocol_hash:
        raise ValueError("Results belong to another frozen protocol; use a new results directory")
    metadata, corpus, queries, arrays = inputs(args)
    if protocol["version"] != 1 or protocol["inputs"] != metadata["inputs"]:
        raise ValueError("Frozen protocol does not match the encoded dataset")
    if (protocol["model"], protocol["revision"]) != (MODEL, REVISION):
        raise ValueError("Frozen protocol uses a different model revision")
    if protocol["embeddings_sha256"] != metadata["embeddings_sha256"] and not args.reproduction:
        raise ValueError("Embedding cache does not match the frozen protocol")
    qrels = load_qrels(args.data, "test")
    train = load_qrels(args.data, "train")
    if set(qrels) & set(train) or {queries[q] for q in qrels} & {queries[q] for q in train}:
        raise ValueError("Development and test queries overlap")
    qids, docids = sorted(qrels), metadata["documents"]
    texts = [queries[qid] for qid in qids]
    started = time.perf_counter()
    lexical = BM25([corpus[docid] for docid in docids])
    lexical_index_seconds = time.perf_counter() - started
    started = time.perf_counter()
    sparse_scores = np.stack([lexical.score(text, **protocol["bm25"]) for text in texts])
    sparse = positions(sparse_scores, docids, qids, sparse=True)
    sparse_seconds = time.perf_counter() - started
    started = time.perf_counter()
    scores = dense_scores(arrays, metadata, qids, protocol["dense_mode"])
    dense = positions(scores, docids, qids)
    dense_search_seconds = time.perf_counter() - started
    methods = {"bm25": rankings(sparse, docids, qids), "bge_small": rankings(dense, docids, qids)}
    methods["weighted_rrf"] = fuse(
        sparse, dense, protocol["rrf"]["dense_weight"], protocol["rrf"]["k"], docids, qids
    )
    if protocol["candidate"] == "router":
        router = Router(**protocol["router"])
        weights = router.predict(features(sparse, dense, sparse_scores, scores, texts))
        methods["candidate"] = fuse(sparse, dense, weights, protocol["rrf"]["k"], docids, qids)
    else:
        methods["candidate"] = methods["weighted_rrf"]
    metrics, per_query = {}, {}
    for name, results in methods.items():
        metrics[name], per_query[name] = evaluate(qrels, results)
    report = {
        "dataset": "BEIR SciFact official test",
        "queries": len(qids),
        "documents": len(docids),
        "protocol_sha256": protocol_hash,
        "calibration_embedding_sha256": protocol["embeddings_sha256"],
        "evaluation_embedding_sha256": metadata["embeddings_sha256"],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "metrics": metrics,
        "comparisons": {
            name: paired_interval(per_query["candidate"], per_query[name])
            for name in ("bm25", "bge_small", "weighted_rrf")
        },
        "cost": {
            "lexical_index_seconds": lexical_index_seconds,
            "sparse_search_and_sort_seconds": sparse_seconds,
            "dense_cached_dot_and_sort_seconds": dense_search_seconds,
            "dense_index_bytes": int(arrays["corpus"].nbytes),
            "encoder": {
                key: value
                for key, value in metadata.items()
                if key not in ("documents", "queries", "inputs")
            },
        },
    }
    args.results.mkdir(parents=True, exist_ok=True)
    write_json(args.results / "evaluation.json", report)
    write_json(args.results / "per_query.json", per_query)
    (args.results / "rankings.json.gz").write_bytes(
        gzip.compress(json.dumps(methods).encode(), mtime=0)
    )
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "tune", "test"))
    parser.add_argument("--data", type=Path, default=Path("data/scifact"))
    parser.add_argument("--cache", type=Path, default=Path("cache"))
    parser.add_argument("--protocol", type=Path, default=Path("protocol.json"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--reproduction",
        action="store_true",
        help="Evaluate frozen settings with a newly encoded cache; record both cache hashes",
    )
    args = parser.parse_args()
    {"prepare": prepare, "tune": tune, "test": test}[args.command](args)


if __name__ == "__main__":
    main()
