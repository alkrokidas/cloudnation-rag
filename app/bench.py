"""Times search and reranker on this machine, so the reranker settings are chosen from numbers.

    docker compose exec api python -m app.bench

Runs the 16 simple questions of the router examples (the System 1 questions) at 512 tokens, where TAU_LOW and
TAU_HIGH were measured, and with cheaper settings, and shows how far the top score moves and how often the grade
changes compared to 512 tokens. No LLM calls, no cost. Takes 1-2 minutes on 2 CPU cores.
Rows tried and rejected on 23/10 (DEPTH 5, int8 weights, 2 threads on 2 vCPUs) are listed in config.py.
"""
import os
import statistics
import time
from collections import Counter

import torch

from app.config import DEPTH, RERANK_BATCH, TAU_HIGH, TAU_LOW
from app.encoders import reranker
from app.graph import zone
from app.router import ROUTER_EXAMPLES
from app.search import search_qdrant

QUESTIONS = [q for q, label in ROUTER_EXAMPLES if label == 0]
DEFAULT_THREADS, VCPUS = torch.get_num_threads(), os.cpu_count()
VARIANTS = [   # (name, max tokens per question + chunk, PyTorch threads); the TAUs were measured at 512 tokens
    (f"512 tokens, {DEFAULT_THREADS} thread(s): baseline", 512, DEFAULT_THREADS),
    (f"256 tokens, {DEFAULT_THREADS} thread(s)", 256, DEFAULT_THREADS),
    (f"256 tokens, all {VCPUS} vCPUs as threads", 256, VCPUS),
]


def timed(fn):
    start = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - start) * 1000


def main():
    print(f"{len(QUESTIONS)} questions, role helpdesk, DEPTH {DEPTH}, batch {RERANK_BATCH}, "
          f"PyTorch threads {DEFAULT_THREADS}, vCPUs {VCPUS}", flush=True)
    candidates, search_ms = {}, []
    for q in QUESTIONS:
        candidates[q], ms = timed(lambda: search_qdrant(q, "helpdesk"))
        search_ms.append(ms)
    print(f"{'embed + Qdrant search':36} median {statistics.median(search_ms):5.0f} ms   slowest {max(search_ms):5.0f} ms")
    reranker.predict([(QUESTIONS[0], t) for t in candidates[QUESTIONS[0]]["embed_text"]])   # warm-up

    baseline = None
    for name, max_tokens, threads in VARIANTS:
        reranker.max_seq_length = max_tokens
        torch.set_num_threads(threads)
        times, tops = [], []
        for q in QUESTIONS:
            pairs = [(q, t) for t in candidates[q]["embed_text"]]
            scores, ms = timed(lambda: reranker.predict(pairs, batch_size=RERANK_BATCH))
            times.append(ms)
            tops.append(float(max(scores)))
        baseline = baseline or tops
        changed = sum(zone(a) != zone(b) for a, b in zip(tops, baseline))
        moved = max(abs(a - b) for a, b in zip(tops, baseline))
        counts = Counter(zone(t) for t in tops)
        print(f"{name:36} median {statistics.median(times):5.0f} ms   slowest {max(times):5.0f} ms   "
              f"score moved <= {moved:4.2f}   grade changed {changed}/{len(QUESTIONS)}   "
              f"(R {counts['Relevant']}, A {counts['Ambiguous']}, I {counts['Irrelevant']})", flush=True)
    print(f"For scale: TAU_LOW {TAU_LOW}, TAU_HIGH {TAU_HIGH}, {TAU_HIGH - TAU_LOW:.2f} apart")


if __name__ == "__main__":
    main()
