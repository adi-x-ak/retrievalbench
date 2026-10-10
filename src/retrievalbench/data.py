"""Pinned BEIR SciFact bytes, explicit splits, and immutable input fingerprints."""

import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
ARCHIVE_SHA256 = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"
FILES = ("corpus.jsonl", "queries.jsonl", "qrels/train.tsv", "qrels/test.tsv")


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(path: Path) -> None:
    with urllib.request.urlopen(URL, timeout=60) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Dataset archive does not match the pinned checksum")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for name in FILES:
            destination = path / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(f"scifact/{name}"))


def load_text(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    corpus = {}
    for line in (path / "corpus.jsonl").read_text().splitlines():
        item = json.loads(line)
        identifier = str(item["_id"])
        if identifier in corpus:
            raise ValueError("Duplicate corpus ID")
        corpus[identifier] = f"{item.get('title', '')} {item['text']}".strip()
    queries = {}
    for line in (path / "queries.jsonl").read_text().splitlines():
        item = json.loads(line)
        identifier = str(item["_id"])
        if identifier in queries:
            raise ValueError("Duplicate query ID")
        queries[identifier] = item["text"]
    return corpus, queries


def load_qrels(path: Path, split: str) -> dict[str, dict[str, int]]:
    if split not in ("train", "test"):
        raise ValueError("Split must be train or test")
    qrels = {}
    with (path / "qrels" / f"{split}.tsv").open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return qrels


def input_hashes(path: Path) -> dict[str, str]:
    return {name: fingerprint(path / name) for name in FILES}


def development_split(queries: dict[str, str], qrels: dict) -> tuple[dict, list[str]]:
    """Exclude exact held-out text duplicates without inspecting held-out labels."""
    reserved_texts = {text for qid, text in queries.items() if qid not in qrels}
    excluded = sorted(qid for qid in qrels if queries[qid] in reserved_texts)
    return {qid: labels for qid, labels in qrels.items() if qid not in excluded}, excluded
