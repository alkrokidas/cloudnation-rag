"""The two local models, loaded once when the app starts (NB2, NB4)."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import EMBEDDER, RERANK_MAX_TOKENS, RERANK_WORKERS, RERANKER

embedder = SentenceTransformer(EMBEDDER)
# The grader thresholds (TAU_LOW, TAU_HIGH) are raw logits, as measured in NB7 (-5 to 6.5).
# Identity is set explicitly, so a library default (sigmoid, 0 to 1) can never rescale the scores silently.
reranker = CrossEncoder(RERANKER, max_length=RERANK_MAX_TOKENS, activation_fn=torch.nn.Identity())

PHYSICAL_CORES = torch.get_num_threads()          # PyTorch's default: one thread per physical core
# On a 1-core machine two at once would fight (7.5 s on 2 vCPUs); a GPU runs the passes one after another anyway.
WORKERS = 1 if torch.cuda.is_available() else min(RERANK_WORKERS, PHYSICAL_CORES)
if WORKERS > 1:
    torch.set_num_threads(1)                      # each pair gets one core of its own
_pools = {}


def _on_device(features):
    """predict() moves the tokens to the model's device itself; calling the model directly, we do it (GPU)."""
    return {k: v.to(reranker.device) if isinstance(v, torch.Tensor) else v for k, v in features.items()}


def _score(features):
    with torch.inference_mode():
        return float(reranker(features)["scores"].reshape(-1)[0])


def rerank(pairs, workers=WORKERS):
    """Raw-logit scores for (question, chunk) pairs, one pair per forward pass (no padding).
    With workers > 1 that many pairs run at the same time, one core each."""
    features = [_on_device(reranker.preprocess([pair])) for pair in pairs]   # tokenize here: not thread-safe
    if workers == 1:
        return np.array([_score(f) for f in features])
    pool = _pools.setdefault(workers, ThreadPoolExecutor(workers))
    return np.array(list(pool.map(_score, features)))


@lru_cache(maxsize=256)
def embed_query(text):
    """e5 expects "query: " in front of questions and "passage: " in front of documents.
    Cached: the router and the search embed the same question."""
    return embedder.encode("query: " + text, normalize_embeddings=True)


def embed_passages(texts):
    return embedder.encode(["passage: " + t for t in texts], normalize_embeddings=True)
