"""Stable rankings, weighted reciprocal-rank fusion, and a small ridge router."""

from dataclasses import dataclass

import numpy as np


def positions(
    scores: np.ndarray, documents: list[str], queries: list[str], *, sparse: bool = False
) -> np.ndarray:
    if scores.shape != (len(queries), len(documents)) or not np.isfinite(scores).all():
        raise ValueError("Scores must be finite and match query/document IDs")
    if documents != sorted(set(documents)) or len(set(queries)) != len(queries):
        raise ValueError("Document IDs must be sorted and unique; query IDs must be unique")
    values = scores.copy()
    columns = {docid: index for index, docid in enumerate(documents)}
    for row, qid in enumerate(queries):
        if qid in columns:
            values[row, columns[qid]] = -np.inf
    order = np.argsort(-values, axis=1, kind="stable")
    ranks = np.empty_like(order)
    ranks[np.arange(len(queries))[:, None], order] = np.arange(len(documents)) + 1
    invalid = ranks > 1000
    if sparse:
        invalid |= values <= 0
    invalid |= ~np.isfinite(values)
    return np.where(invalid, 0, ranks)


def rankings(ranks: np.ndarray, documents: list[str], queries: list[str], limit: int = 100) -> dict:
    result = {}
    for qid, row in zip(queries, ranks, strict=True):
        order = np.argsort(np.where(row > 0, row, len(documents) + 1), kind="stable")
        result[qid] = [documents[index] for index in order[:limit] if row[index] > 0]
    return result


def rrf(sparse: np.ndarray, dense: np.ndarray, weight: float | np.ndarray, k: float) -> np.ndarray:
    weights = np.asarray(weight)
    if sparse.shape != dense.shape or k <= 0 or not np.isfinite(weights).all():
        raise ValueError("Fusion requires matching ranks, finite weights, and k > 0")
    if np.any((weights < 0) | (weights > 1)):
        raise ValueError("Dense weights must be between zero and one")
    if weights.ndim:
        if weights.shape != (len(sparse),):
            raise ValueError("One weight per query is required")
        weights = weights[:, None]
    return (1 - weights) * np.where(sparse > 0, 1 / (k + sparse), 0) + weights * np.where(
        dense > 0, 1 / (k + dense), 0
    )


def features(
    sparse: np.ndarray,
    dense: np.ndarray,
    sparse_scores: np.ndarray,
    dense_scores: np.ndarray,
    queries: list[str],
) -> np.ndarray:
    s = np.sort(sparse_scores, axis=1)[:, -2:]
    d = np.sort(dense_scores, axis=1)[:, -2:]
    return np.column_stack(
        [
            np.log1p([len(query.split()) for query in queries]),
            s[:, -1],
            (s[:, -1] - s[:, -2]) / np.maximum(s[:, -1], 1e-12),
            d[:, -1],
            d[:, -1] - d[:, -2],
            np.mean((sparse > 0) & (sparse <= 10) & (dense > 0) & (dense <= 10), axis=1)
            * sparse.shape[1]
            / 10,
            np.mean((sparse > 0) & (sparse <= 100) & (dense > 0) & (dense <= 100), axis=1)
            * sparse.shape[1]
            / 100,
        ]
    )


@dataclass
class Router:
    mean: list[float]
    scale: list[float]
    coefficients: list[list[float]]
    choices: list[float]

    @classmethod
    def fit(
        cls, x: np.ndarray, utilities: np.ndarray, choices: list[float], penalty: float
    ) -> "Router":
        if penalty <= 0 or len(x) == 0 or utilities.shape != (len(x), len(choices)):
            raise ValueError("Router needs aligned utilities and positive regularization")
        mean, scale = x.mean(0), np.maximum(x.std(0), 1e-12)
        design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
        regularization = np.eye(design.shape[1]) * penalty
        regularization[0, 0] = 0
        coefficients = np.linalg.solve(design.T @ design + regularization, design.T @ utilities)
        return cls(mean.tolist(), scale.tolist(), coefficients.tolist(), choices)

    def predict(self, x: np.ndarray) -> np.ndarray:
        design = np.column_stack([np.ones(len(x)), (x - self.mean) / self.scale])
        scores = design @ np.array(self.coefficients)
        if not np.isfinite(scores).all():
            raise ValueError("Router predictions must be finite")
        return np.array(self.choices)[scores.argmax(axis=1)]
