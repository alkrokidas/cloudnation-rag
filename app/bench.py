"""Times search and reranker on this machine, so the reranker settings are chosen from numbers.

    docker compose exec api python -m app.bench

Runs the 16 simple questions of the router examples (the System 1 questions) at 512 tokens, where TAU_LOW and
TAU_HIGH were measured, and with faster settings, and shows how far the top score moves and how often the grade
changes compared to 512 tokens. No LLM calls, no cost. Takes 1-2 minutes.
Rows tried and rejected on 23/10 (DEPTH 5, int8 weights, 2 threads on 2 vCPUs) are listed in config.py.
"""
import os
import platform
import statistics
import time
from collections import Counter

import torch

from app.config import DEPTH, TAU_HIGH, TAU_LOW
from app.encoders import PHYSICAL_CORES, WORKERS, rerank, reranker
from app.graph import zone
from app.router import ROUTER_EXAMPLES
from app.search import search_qdrant

QUESTIONS = [q for q, label in ROUTER_EXAMPLES if label == 0]
VARIANTS = [   # (name, max tokens per question + chunk, pairs at the same time, bfloat16); first row = baseline
    ("512 tokens: baseline", 512, 1, False),
    ("256 tokens", 256, 1, False),
    ("256 tokens, 2 pairs at once", 256, 2, False),
    ("256 tokens, bf16", 256, 1, True),
    ("256 tokens, 2 at once + bf16", 256, 2, True),
    ("192 tokens, 2 at once + bf16", 192, 2, True),
]


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            info = f.read()
    except OSError:
        return platform.processor(), "?"
    name = next((line.split(":", 1)[1].strip() for line in info.splitlines() if line.startswith("model name")), "?")
    return name, "yes" if "avx512_bf16" in info or "amx_bf16" in info else "no"


def timed(fn):
    start = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - start) * 1000


def main():
    name, bf16_hw = cpu_name()
    print(f"CPU: {name}, {os.cpu_count()} vCPUs, {PHYSICAL_CORES} physical cores, bf16 in hardware: {bf16_hw}")
    print(f"{len(QUESTIONS)} questions, role helpdesk, DEPTH {DEPTH}, app uses {WORKERS} pair(s) at once", flush=True)
    candidates, search_ms = {}, []
    for q in QUESTIONS:
        candidates[q], ms = timed(lambda: search_qdrant(q, "helpdesk"))
        search_ms.append(ms)
    print(f"{'embed + Qdrant search':30} median {statistics.median(search_ms):5.0f} ms   slowest {max(search_ms):5.0f} ms")

    baseline = None
    for label, max_tokens, workers, bf16 in VARIANTS:
        if workers > PHYSICAL_CORES:
            print(f"{label:30} skipped: needs {workers} physical cores")
            continue
        reranker.max_seq_length = max_tokens
        torch.set_num_threads(1 if workers > 1 else PHYSICAL_CORES)
        rerank([(QUESTIONS[0], "warm-up")], workers, bf16)
        times, tops = [], []
        for q in QUESTIONS:
            pairs = [(q, t) for t in candidates[q]["embed_text"]]
            scores, ms = timed(lambda: rerank(pairs, workers, bf16))
            times.append(ms)
            tops.append(float(max(scores)))
        baseline = baseline or tops
        changed = sum(zone(a) != zone(b) for a, b in zip(tops, baseline))
        moved = max(abs(a - b) for a, b in zip(tops, baseline))
        counts = Counter(zone(t) for t in tops)
        print(f"{label:30} median {statistics.median(times):5.0f} ms   slowest {max(times):5.0f} ms   "
              f"score moved <= {moved:4.2f}   grade changed {changed}/{len(QUESTIONS)}   "
              f"(R {counts['Relevant']}, A {counts['Ambiguous']}, I {counts['Irrelevant']})", flush=True)
    print(f"For scale: TAU_LOW {TAU_LOW}, TAU_HIGH {TAU_HIGH}, {TAU_HIGH - TAU_LOW:.2f} apart")


if __name__ == "__main__":
    main()
