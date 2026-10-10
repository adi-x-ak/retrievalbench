"""Pinned BGE-small CLS embeddings; no remote model code is executed."""

import time

import numpy as np

MODEL = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
INSTRUCTION = "Represent this sentence for searching relevant passages: "


def encode(documents: list[str], queries: list[str], device: str) -> tuple[dict, dict]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    model = (
        AutoModel.from_pretrained(
            MODEL, revision=REVISION, trust_remote_code=False, use_safetensors=True
        )
        .eval()
        .to(device)
    )

    @torch.inference_mode()
    def batches(texts: list[str]) -> np.ndarray:
        vectors = []
        for start in range(0, len(texts), 16):
            inputs = tokenizer(
                texts[start : start + 16],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            vectors.append(
                torch.nn.functional.normalize(model(**inputs).last_hidden_state[:, 0], dim=-1)
                .cpu()
                .numpy()
            )
            if start % 512 == 0:
                print(f"Encoded {min(start + 16, len(texts))}/{len(texts)} texts", flush=True)
        return np.concatenate(vectors)

    started = time.perf_counter()
    corpus = batches(documents)
    corpus_seconds = time.perf_counter() - started
    started = time.perf_counter()
    plain = batches(queries)
    instructed = batches([INSTRUCTION + text for text in queries])
    query_seconds = time.perf_counter() - started
    if not all(np.isfinite(vectors).all() for vectors in (corpus, plain, instructed)):
        raise ValueError("Encoder produced nonfinite embeddings")
    return {"corpus": corpus, "plain": plain, "instructed": instructed}, {
        "model": MODEL,
        "revision": REVISION,
        "device": device,
        "max_length": 512,
        "batch_size": 16,
        "cpu_threads": 4,
        "corpus_encoding_seconds": corpus_seconds,
        "query_encoding_seconds_both_modes": query_seconds,
        "dimension": int(corpus.shape[1]),
    }
