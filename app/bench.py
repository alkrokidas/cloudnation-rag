"""Times search and reranker on this machine, so the reranker settings are chosen from numbers.

    docker compose exec api python -m app.bench

Runs the 16 simple questions of the router examples (the System 1 questions) with the current settings and
cheaper ones, and shows how far the top score moves and how often the grade changes compared to now, because
TAU_LOW and TAU_HIGH were measured with the current settings. No LLM calls, no cost. Takes 1-3 minutes on 2 CPU cores.
"""
import statistics
import time
import warnings
from collections import Counter

import torch

from app.config import DEPTH, TAU_HIGH, TAU_LOW
from app.encoders import reranker
from app.graph import zone
from app.router import ROUTER_EXAMPLES
from app.search import search_qdrant

QUESTIONS = [q for q, label in ROUTER_EXAMPLES if label == 0]
VARIANTS = [   # (name, candidates the reranker reads, max tokens per question + chunk, pairs per batch, int8)
    ("now: DEPTH 10, 512 tokens, batch 32", DEPTH, 512, 32, False),
    ("batch 1: no padding, same scores", DEPTH, 512, 1, False),
    ("256 tokens: long chunks cut", DEPTH, 256, 32, False),
    ("DEPTH 5: half the candidates", 5, 512, 32, False),
    ("int8 weights + batch 1", DEPTH, 512, 1, True),       # int8 rows last: quantizing changes the model in place
    ("int8 + batch 1 + DEPTH 5", 5, 512, 1, True),
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
    for name, depth, max_tokens, batch, int8 in VARIANTS:
        reranker.max_seq_length = max_tokens
        if int8:      # 8-bit weights for the linear layers: smaller and faster on CPU, scores move a little
            with warnings.catch_warnings():     # PyTorch plans to move this to its torchao package; it works in 2.14
                warnings.simplefilter("ignore")
                torch.ao.quantization.quantize_dynamic(reranker, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
        times, tops = [], []
        for q in QUESTIONS:
            pairs = [(q, t) for t in candidates[q]["embed_text"].head(depth)]
            scores, ms = timed(lambda: reranker.predict(pairs, batch_size=batch))
            times.append(ms)
            tops.append(float(max(scores)))
        baseline = baseline or tops
        grades = [zone(t) for t in tops]
        changed = sum(zone(a) != zone(b) for a, b in zip(tops, baseline))
        moved = max(abs(a - b) for a, b in zip(tops, baseline))
        counts = Counter(grades)
        print(f"{name:38} median {statistics.median(times):5.0f} ms   slowest {max(times):5.0f} ms   "
              f"top score moved up to {moved:4.2f}   grade changed {changed}/{len(QUESTIONS)}   "
              f"(R {counts['Relevant']}, A {counts['Ambiguous']}, I {counts['Irrelevant']})", flush=True)
    print(f"For scale: TAU_LOW {TAU_LOW}, TAU_HIGH {TAU_HIGH}, {TAU_HIGH - TAU_LOW:.2f} apart")


if __name__ == "__main__":
    main()
