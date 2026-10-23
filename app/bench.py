"""Times search, reranker and the API connection on this machine, so the settings are chosen from numbers.

    docker compose exec api python -m app.bench

Runs the 16 simple questions of the router examples (the System 1 questions) at 512 tokens, where TAU_LOW and
TAU_HIGH were measured, and with faster settings, and shows how far the top score moves and how often the grade
changes compared to 512 tokens. Then times free count_tokens calls to the API on a new and on a reused
connection. No LLM answers, no cost. Takes 1-2 minutes.
Rows tried and rejected on 23/10 (DEPTH 5, int8 weights, 2 threads on 2 vCPUs, bf16, 192 tokens) are in config.py.
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
from app.llm import count_tokens
from app.router import ROUTER_EXAMPLES
from app.search import search_qdrant

QUESTIONS = [q for q, label in ROUTER_EXAMPLES if label == 0]
VARIANTS = [   # (name, max tokens per question + chunk, pairs at the same time); first row = baseline
    ("512 tokens: baseline", 512, 1),
    ("256 tokens", 256, 1),
    ("256 tokens, 2 pairs at once", 256, 2),
]


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            info = f.read()
    except OSError:
        return platform.processor()
    return next((line.split(":", 1)[1].strip() for line in info.splitlines() if line.startswith("model name")), "?")


def timed(fn):
    start = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - start) * 1000


def api_connection(runs=3):
    """The SDK closes a connection after 5 s without use, so after 6 s the next call opens a new one."""
    try:
        timed(count_tokens)
    except Exception as error:
        print(f"{'API connection':30} skipped: {type(error).__name__} (is ANTHROPIC_API_KEY set?)")
        return
    new, reused = [], []
    for _ in range(runs):
        time.sleep(6)
        new.append(timed(count_tokens)[1])
        reused.append(timed(count_tokens)[1])
    new_ms, reused_ms = statistics.median(new), statistics.median(reused)
    print(f"{'API call, new connection':30} median {new_ms:5.0f} ms")
    print(f"{'API call, reused connection':30} median {reused_ms:5.0f} ms   "
          f"-> opening the connection costs ~{new_ms - reused_ms:.0f} ms (the app now does it during the search)")


def main():
    print(f"CPU: {cpu_name()}, {os.cpu_count()} vCPUs, {PHYSICAL_CORES} physical cores")
    print(f"{len(QUESTIONS)} questions, role helpdesk, DEPTH {DEPTH}, app uses {WORKERS} pair(s) at once", flush=True)
    candidates, search_ms = {}, []
    for q in QUESTIONS:
        candidates[q], ms = timed(lambda: search_qdrant(q, "helpdesk"))
        search_ms.append(ms)
    print(f"{'embed + Qdrant search':30} median {statistics.median(search_ms):5.0f} ms   slowest {max(search_ms):5.0f} ms")

    baseline = None
    for label, max_tokens, workers in VARIANTS:
        if workers > PHYSICAL_CORES:
            print(f"{label:30} skipped: needs {workers} physical cores")
            continue
        reranker.max_seq_length = max_tokens
        torch.set_num_threads(1 if workers > 1 else PHYSICAL_CORES)
        rerank([(QUESTIONS[0], "warm-up")], workers)
        times, tops = [], []
        for q in QUESTIONS:
            pairs = [(q, t) for t in candidates[q]["embed_text"]]
            scores, ms = timed(lambda: rerank(pairs, workers))
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
    api_connection()


if __name__ == "__main__":
    main()
