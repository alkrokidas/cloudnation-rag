"""Times search and reranker on this machine, so the reranker settings are chosen from numbers.

    docker compose exec api python -m app.bench

Runs the 16 simple questions of the router examples (the System 1 questions) with the current settings and
three cheaper ones, and counts how often the grade changes compared to now, because TAU_LOW and TAU_HIGH
were measured with the current settings. No LLM calls, no cost. Takes 1-3 minutes on 2 CPU cores.
"""
import statistics
import time
from collections import Counter

import torch

from app.config import DEPTH
from app.encoders import reranker
from app.graph import zone
from app.router import ROUTER_EXAMPLES
from app.search import search_qdrant

QUESTIONS = [q for q, label in ROUTER_EXAMPLES if label == 0]
VARIANTS = [   # (name, candidates the reranker reads, max tokens per question + chunk, pairs per batch)
    ("now: DEPTH 10, 512 tokens, batch 32", DEPTH, 512, 32),
    ("batch 1: no padding, same scores", DEPTH, 512, 1),
    ("256 tokens: long chunks cut", DEPTH, 256, 32),
    ("DEPTH 5: half the candidates", 5, 512, 32),
]


def timed(fn):
    start = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - start) * 1000


def main():
    print(f"{len(QUESTIONS)} questions, role helpdesk, PyTorch threads: {torch.get_num_threads()}", flush=True)
    candidates, search_ms = {}, []
    for q in QUESTIONS:
        candidates[q], ms = timed(lambda: search_qdrant(q, "helpdesk"))
        search_ms.append(ms)
    print(f"{'embed + Qdrant search':38} median {statistics.median(search_ms):5.0f} ms   slowest {max(search_ms):5.0f} ms")
    reranker.predict([(QUESTIONS[0], t) for t in candidates[QUESTIONS[0]]["embed_text"]])   # warm-up

    baseline = None
    for name, depth, max_tokens, batch in VARIANTS:
        reranker.max_seq_length = max_tokens
        times, grades = [], []
        for q in QUESTIONS:
            pairs = [(q, t) for t in candidates[q]["embed_text"].head(depth)]
            scores, ms = timed(lambda: reranker.predict(pairs, batch_size=batch))
            times.append(ms)
            grades.append(zone(float(max(scores))))
        baseline = baseline or grades
        changed = sum(a != b for a, b in zip(grades, baseline))
        counts = Counter(grades)
        print(f"{name:38} median {statistics.median(times):5.0f} ms   slowest {max(times):5.0f} ms   "
              f"grade changed {changed}/{len(QUESTIONS)}   (Relevant {counts['Relevant']}, "
              f"Ambiguous {counts['Ambiguous']}, Irrelevant {counts['Irrelevant']})", flush=True)
    reranker.max_seq_length = 512


if __name__ == "__main__":
    main()
