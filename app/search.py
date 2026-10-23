"""Hybrid search inside Qdrant with the role filter on every path (NB3, NB4, NB6)."""
import re
import zlib
from collections import Counter

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models

from app.config import COLLECTION, DEPTH, MAX_CHUNKS_PER_DOC, PREFETCH, QDRANT_URL, ROLES, RRF_K, TOP_K
from app.encoders import embed_query, reranker

qdrant = QdrantClient(QDRANT_URL) if QDRANT_URL == ":memory:" else QdrantClient(url=QDRANT_URL)

ECLI_PATTERN = re.compile(r"ECLI:[A-Z]{2}:[A-Z0-9]{1,7}:\d{4}:[A-Z0-9.]{1,25}", re.IGNORECASE)


def tokenize(text):
    """Words, keeping numbers like 3.114, 71,867 and 2026/01 in one piece (NB3)."""
    return re.findall(r"\w+(?:[.,:/]\w+)*", text.lower())


def token_id(token):
    """BM25 needs a number per word: a stable hash, so ingest and search agree without a shared vocabulary."""
    return zlib.crc32(token.encode()) & 0x7FFFFFFF


def sparse_query(question):
    counts = Counter(token_id(t) for t in tokenize(question))
    return models.SparseVector(indices=list(counts), values=[float(v) for v in counts.values()])


def role_filter(role):
    """WHERE classification IN (...). The only place a filter is built, so no path can forget it (NB6)."""
    return models.Filter(must=[models.FieldCondition(key="classification", match=models.MatchAny(any=ROLES[role]))])


def search_qdrant(question, role, k=DEPTH):
    """Dense + BM25, fused with RRF inside Qdrant. The filter runs in both prefetches, before ranking."""
    flt = role_filter(role)
    prefetch = [models.Prefetch(query=embed_query(question).tolist(), using="dense", filter=flt, limit=PREFETCH)]
    q_sparse = sparse_query(question)
    if q_sparse.indices:
        prefetch.append(models.Prefetch(query=q_sparse, using="bm25", filter=flt, limit=PREFETCH))
    result = qdrant.query_points(COLLECTION, prefetch=prefetch, query=models.RrfQuery(rrf=models.Rrf(k=RRF_K)),
                                 query_filter=flt, limit=k, with_payload=True)
    return pd.DataFrame([p.payload for p in result.points])


def lookup_ecli(ecli, role, k=TOP_K):
    """Exact lookup by case number, like a primary key, with the same role filter."""
    flt = models.Filter(must=role_filter(role).must + [models.FieldCondition(key="ecli", match=models.MatchValue(value=ecli))])
    points, _ = qdrant.scroll(COLLECTION, scroll_filter=flt, limit=k, with_payload=True)
    return pd.DataFrame([p.payload for p in points])


def diverse_order(candidates, scores, k=TOP_K, per_doc=MAX_CHUNKS_PER_DOC):
    """Best first, but at most per_doc chunks from one document, so one long ruling cannot fill all k places."""
    picked, taken = [], Counter()
    for i in np.argsort(-scores):
        parent = candidates.iloc[i]["parent_id"]
        if taken[parent] < per_doc:
            picked.append(i)
            taken[parent] += 1
        if len(picked) == k:
            break
    return picked


def retrieve_scored(search_text, question, role):
    """Returns (chunks, grade score, exact).

    search_text: what we search for (changes on a retry).
    question: what the reranker scores against (always the user's own question).
    The score is the reranker score of the best chunk; it is None for an exact ECLI match.
    """
    match = ECLI_PATTERN.search(question)
    if match:
        hits = lookup_ecli(match.group(0).upper(), role)
        if len(hits):
            return hits, None, True
    candidates = search_qdrant(search_text, role)
    if candidates.empty:
        return candidates, float("-inf"), False
    scores = reranker.predict([(question, t) for t in candidates["embed_text"]])
    order = diverse_order(candidates, scores)
    return candidates.iloc[order].reset_index(drop=True), float(scores[order[0]]), False
