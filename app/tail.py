"""Haiku's time to first token, with and without a hedged call, so the hedge delay is chosen from numbers.

    python -m app.tail                    # 80 rounds, ~6 minutes, ~$0.10
    python -m app.tail --rounds 20        # a quick look

Why: over 18 answers the golden set's System 1 p95 swung from 1.3 to 3.5 s between runs of the same code (27/10),
decided by one or two slow first tokens from the API. A p95 needs more calls than that.

Every round sends the demo question's real prompt (its sources, the rules, the language line) once per variant: no
hedge, and a second call after each delay in --delays. The order of the variants turns every round, so all of them
meet the API at the same moments. Only the first token matters here, so max_tokens is small; it does not change when
the first token comes. Calls start 1.5 s apart (~40 per minute, below a low rate limit), and the SDK does not retry,
so a rate-limit error is counted instead of showing up as a slow call.
"""
import argparse
import time

import anthropic
import numpy as np

from app import ingest, llm
from app.config import HEDGE_AFTER_MS, SYSTEM1_MODEL
from app.llm import ask_llm, system_prompt, user_message
from app.main import warm_up_llm
from app.search import retrieve_scored

QUESTION = "Mag ik de kosten van mijn werkkamer thuis aftrekken?"   # the demo question
SPACING_S = 1.5
MAX_TOKENS = 16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--delays", default="700,1000", help="hedge delays to try, in ms, comma-separated")
    args = parser.parse_args()
    delays = {"no hedge": None} | {f"hedge after {d} ms": int(d) for d in args.delays.split(",")}
    names = list(delays)

    ingest.ensure_collection()
    llm._client = anthropic.Anthropic(max_retries=0)      # a retry would look like a slow first token
    warm_up_llm()
    rows, _, _ = retrieve_scored(QUESTION, QUESTION, "helpdesk")
    system, user = system_prompt("nl", 5), user_message(rows, QUESTION, "nl")
    minutes = args.rounds * len(names) * SPACING_S / 60
    print(f"{args.rounds} rounds x {len(names)} variants, calls {SPACING_S} s apart, ~{minutes:.0f} min", flush=True)

    ttft = {name: [] for name in names}
    hedged, errors, cost = dict.fromkeys(names, 0), dict.fromkeys(names, 0), dict.fromkeys(names, 0.0)
    next_start = time.perf_counter()
    for r in range(args.rounds):
        for name in names[r % len(names):] + names[:r % len(names)]:
            time.sleep(max(0.0, next_start - time.perf_counter()))
            next_start = time.perf_counter() + SPACING_S
            HEDGE_AFTER_MS[SYSTEM1_MODEL] = delays[name]
            try:
                _, stats = ask_llm(SYSTEM1_MODEL, system, user, max_tokens=MAX_TOKENS)
            except anthropic.APIError as error:
                errors[name] += 1
                print(f"  {name}: {type(error).__name__}", flush=True)
                continue
            ttft[name].append(stats["ttft_ms"])
            hedged[name] += stats["hedged"]
            cost[name] += stats["cost"]
        if (r + 1) % 10 == 0:
            print(f"  round {r + 1}/{args.rounds}", flush=True)
    HEDGE_AFTER_MS.pop(SYSTEM1_MODEL, None)

    print(f"\nTime to first token of Haiku (ms), {QUESTION!r}:")
    print(f"{'':22} {'n':>4} {'p50':>6} {'p90':>6} {'p95':>6} {'p99':>6} {'max':>6}  hedged  errors  $ per 1000")
    for name in names:
        ms = np.array(ttft[name])
        if not len(ms):
            print(f"{name:22} no answers")
            continue
        p50, p90, p95, p99 = np.percentile(ms, [50, 90, 95, 99])
        print(f"{name:22} {len(ms):4} {p50:6.0f} {p90:6.0f} {p95:6.0f} {p99:6.0f} {ms.max():6.0f}"
              f"  {hedged[name] / len(ms):6.0%}  {errors[name]:6}  {cost[name] / len(ms) * 1000:10.2f}")
    print("Add ~0.15 s for route and retrieve on a T4 to get the first word on the server.")


if __name__ == "__main__":
    main()
