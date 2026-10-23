"""The two local models, loaded once when the app starts (NB2, NB4)."""
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import EMBEDDER, RERANKER

embedder = SentenceTransformer(EMBEDDER)
# The grader thresholds (TAU_LOW, TAU_HIGH) are raw logits, as measured in NB7 (-5 to 6.5).
# Identity is set explicitly, so a library default (sigmoid, 0 to 1) can never rescale the scores silently.
reranker = CrossEncoder(RERANKER, max_length=512, activation_fn=torch.nn.Identity())


def embed_query(text):
    """e5 expects "query: " in front of questions and "passage: " in front of documents."""
    return embedder.encode("query: " + text, normalize_embeddings=True)


def embed_passages(texts):
    return embedder.encode(["passage: " + t for t in texts], normalize_embeddings=True)
