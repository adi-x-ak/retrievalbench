"""BM25 with an inspectable inverted index and explicit English analysis."""

import re
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
import Stemmer

STOP_WORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the their "
    "then there these they this to was will with".split()
)
STEMMER = Stemmer.Stemmer("english")


@lru_cache(maxsize=100_000)
def stem(word: str) -> str:
    return STEMMER.stemWord(word)


def tokenize(text: str) -> list[str]:
    return [stem(word) for word in re.findall(r"\w+", text.lower()) if word not in STOP_WORDS]


class BM25:
    def __init__(self, documents: list[str]) -> None:
        if not documents:
            raise ValueError("The corpus must contain documents")
        postings = defaultdict(list)
        lengths = []
        for index, text in enumerate(documents):
            counts = Counter(tokenize(text))
            lengths.append(sum(counts.values()))
            for term, count in counts.items():
                postings[term].append((index, count))
        self.lengths = np.array(lengths, dtype=np.float64)
        self.average_length = self.lengths.mean()
        if self.average_length == 0:
            raise ValueError("The analyzed corpus must contain terms")
        self.postings = {
            term: (
                np.array([i for i, _ in rows]),
                np.array([tf for _, tf in rows], dtype=np.float64),
            )
            for term, rows in postings.items()
        }

    def score(self, query: str, k1: float = 1.2, b: float = 0.75) -> np.ndarray:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Require k1 > 0 and 0 <= b <= 1")
        result = np.zeros(len(self.lengths), dtype=np.float64)
        normalization = k1 * (1 - b + b * self.lengths / self.average_length)
        for term, frequency in Counter(tokenize(query)).items():
            if term not in self.postings:
                continue
            indices, counts = self.postings[term]
            df = len(indices)
            idf = np.log1p((len(self.lengths) - df + 0.5) / (df + 0.5))
            result[indices] += (
                frequency * idf * counts * (k1 + 1) / (counts + normalization[indices])
            )
        return result
