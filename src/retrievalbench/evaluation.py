"""TREC metrics, complete query coverage, and paired bootstrap uncertainty."""

import numpy as np
import pytrec_eval


def evaluate(qrels: dict, rankings: dict) -> tuple[dict, dict]:
    if set(qrels) != set(rankings):
        raise ValueError("Results must cover every labeled query exactly once")
    # Strictly decreasing scores preserve our declared ID tie-break in TREC evaluation.
    runs = {
        qid: {docid: float(len(order) - rank) for rank, docid in enumerate(order) if docid != qid}
        for qid, order in rankings.items()
    }
    if any(len(set(order)) != len(order) for order in rankings.values()):
        raise ValueError("A ranking cannot contain duplicate documents")
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.100"})
    scores = evaluator.evaluate(runs)
    per_query = {qid: scores.get(qid, {"ndcg_cut_10": 0.0, "recall_100": 0.0}) for qid in qrels}
    if not qrels:
        raise ValueError("Evaluation requires labeled queries")
    aggregate = {
        name: float(np.mean([row[name] for row in per_query.values()]))
        for name in ("ndcg_cut_10", "recall_100")
    }
    return aggregate, per_query


def paired_interval(left: dict, right: dict, seed: int = 42, samples: int = 10_000) -> dict:
    if not left or set(left) != set(right) or samples <= 0:
        raise ValueError("Bootstrap needs matching nonempty query sets and positive samples")
    differences = np.array(
        [left[qid]["ndcg_cut_10"] - right[qid]["ndcg_cut_10"] for qid in sorted(left)]
    )
    generator = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for start in range(0, samples, 100):
        indices = generator.integers(
            len(differences), size=(min(100, samples - start), len(differences))
        )
        estimates[start : start + len(indices)] = differences[indices].mean(axis=1)
    return {
        "mean_delta": float(differences.mean()),
        "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "samples": samples,
        "seed": seed,
    }
