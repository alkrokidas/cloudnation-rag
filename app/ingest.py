"""Loads the chunks into Qdrant. Runs once before the API starts: python -m app.ingest"""
import json
import time
from collections import Counter

import numpy as np
import pandas as pd
from qdrant_client import models

from app.config import CHUNKS_CSV, COLLECTION, MOCK_DOCS_JSON, QDRANT_URL
from app.encoders import embed_passages, embedder
from app.search import qdrant, token_id, tokenize

K1, B = 1.5, 0.75   # standard BM25 constants (NB3)


def load_chunks():
    """Public law and rulings from NB3, plus the Dutch FICTIEF internal and FIOD documents from NB6."""
    base = pd.read_csv(CHUNKS_CSV, dtype=str, keep_default_na=False)
    base = base[base["doc_type"].isin(["legislation", "ruling"])]   # the old English mock docs of NB1 stay out
    rows = []
    with open(MOCK_DOCS_JSON, encoding="utf-8") as f:
        for doc in json.load(f):
            header = f"[{doc['title']}]"
            for n, text in enumerate(doc["texts"], start=1):
                rows.append({"chunk_id": f"{doc['parent_id']}-{n}", "doc_type": doc["doc_type"], "law": doc["title"],
                             "year": "2026", "parent_id": doc["parent_id"], "classification": doc["classification"],
                             "context_header": header, "text": text, "embed_text": f"{header} {text}",
                             "n_chars": str(len(text)), "ecli": ""})
    chunks = pd.concat([base, pd.DataFrame(rows)], ignore_index=True).fillna("")
    chunks["ref"] = chunks["ecli"].where(chunks["ecli"] != "",
                                         chunks["article"].where(chunks["article"] != "", chunks["parent_id"]))
    return chunks


def sparse_doc(text, avg_len):
    """BM25 weight per word of a chunk; Qdrant multiplies it by the IDF (Modifier.IDF) at search time."""
    tokens = tokenize(text)
    weights = {}
    for token, tf in Counter(tokens).items():
        i = token_id(token)
        weights[i] = weights.get(i, 0) + tf * (K1 + 1) / (tf + K1 * (1 - B + B * len(tokens) / avg_len))
    return models.SparseVector(indices=list(weights), values=list(weights.values()))


def build():
    chunks = load_chunks()
    avg_len = np.mean([len(tokenize(t)) for t in chunks["embed_text"]])
    if qdrant.collection_exists(COLLECTION):
        qdrant.delete_collection(COLLECTION)          # rebuild from the CSV every time: the CSV is the source of truth
    qdrant.create_collection(
        COLLECTION,
        vectors_config={"dense": models.VectorParams(size=embedder.get_embedding_dimension(),
                                                     distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)})
    for field in ("classification", "ecli"):          # indexes for the filters, like CREATE INDEX in SQL
        qdrant.create_payload_index(COLLECTION, field, models.PayloadSchemaType.KEYWORD)
    vectors = embed_passages(chunks["embed_text"].tolist())
    qdrant.upsert(COLLECTION, points=[
        models.PointStruct(id=i, vector={"dense": vectors[i].tolist(), "bm25": sparse_doc(row["embed_text"], avg_len)},
                           payload=row.to_dict())
        for i, row in chunks.iterrows()])
    return chunks


def ensure_collection():
    """Without Docker there is no ingest step: Qdrant runs inside the process and starts empty (tests, Colab)."""
    if not qdrant.collection_exists(COLLECTION):
        build()


def wait_for_qdrant(seconds=60):
    """In Docker the Qdrant container may still be starting when this one runs."""
    print(f"Waiting for Qdrant at {QDRANT_URL} ...", flush=True)
    deadline = time.monotonic() + seconds          # a clock, not a counter: one try can hang for seconds
    while True:
        try:
            qdrant.get_collections()
            return
        except Exception as error:
            if time.monotonic() > deadline:
                raise RuntimeError(f"Qdrant at {QDRANT_URL} did not answer within {seconds} s: {error}") from error
            time.sleep(1)


if __name__ == "__main__":
    wait_for_qdrant()
    chunks = build()
    counts = chunks["classification"].value_counts()
    print(f"{len(chunks)} chunks in Qdrant:", {level: int(n) for level, n in counts.items()})
