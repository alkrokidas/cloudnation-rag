"""Runs the golden set through the whole system and checks the numbers a new release must keep (Module 5).

    docker compose exec api python -m app.evaluate              # search, grade, answers and their checks (~$0.30)
    docker compose exec api python -m app.evaluate --no-judge   # the same without the paid judge (~$0.10)
    docker compose exec api python -m app.evaluate --search     # search and grade only: no LLM, no cost, ~30 s

Exits with code 1 when a gate in config.py fails, so a CI pipeline can block the release.
"""
import argparse
import sys

import numpy as np
import pandas as pd

from app import ingest
from app.config import GATES, ROLES
from app.golden import GOLDEN, LEAK_TESTS, OUT_OF_SCOPE, ref, right_sources
from app.graph import ask, zone
from app.judge import JUDGE_LLM, bad_citations, judge, language_ok
from app.llm import REFUSAL_MESSAGE
from app.main import warm_up, warm_up_llm
from app.search import retrieve_scored, search_qdrant


def run(question, expected, group, llm, use_judge):
    if llm:
        result = ask(question, "helpdesk")
        rows, grade = result["rows"], result["grade"]
    else:
        rows, score, exact = retrieve_scored(question, question, "helpdesk")
        grade, result = "Relevant" if exact else zone(score), None
    refs = [ref(row) for _, row in rows.iterrows()]
    right = right_sources(expected) if expected else set()
    rank = next((i for i, r in enumerate(refs, start=1) if r in right), None)
    refused = result["answer"] == REFUSAL_MESSAGE if result else None
    checks = {"lang ok": None, "cites ok": None, "wrong claims": None, "answers": None, "judge cost": 0.0, "claims": [],
              "bad cites": [], "answer": result["answer"] if result else ""}
    if result and not refused and expected:          # an answer to an answerable question: check the answer itself
        answer = result["answer"]
        bad = bad_citations(answer, rows)
        checks.update({"lang ok": language_ok(answer, result["language"]), "cites ok": not bad, "bad cites": bad})
        if use_judge:
            verdict, checks["judge cost"] = judge(question, rows, answer)
            if verdict:
                checks.update({"wrong claims": len(verdict["wrong_claims"]), "claims": verdict["wrong_claims"],
                               "answers": verdict["answers_the_question"]})
    return {"group": group, "grade": grade, "rank": rank,
            "system": result["system"] if result else "", "refused": refused,
            "first word ms": round(result["ttft_ms"]) if result else None,
            **checks, "cost": result["cost"] if result else 0.0, "question": question}


def leaks():
    """NB6: chunks outside the role among all candidates of the leak questions, and whether fiod finds FIOD."""
    outside, fiod_found = 0, False
    for role, allowed in ROLES.items():
        for question in LEAK_TESTS:
            found = search_qdrant(question, role)["classification"]
            outside += int((~found.isin(allowed)).sum())
            fiod_found |= role == "fiod" and bool((found == "fiod").any())
    return outside, fiod_found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", action="store_true", help="search and grade only, no LLM calls")
    parser.add_argument("--no-judge", action="store_true", help="skip the paid judge of the answers")
    args = parser.parse_args()
    llm, use_judge = not args.search, not args.search and not args.no_judge
    ingest.ensure_collection()
    warm_up(runs=1)
    if llm:
        warm_up_llm()

    rows = [run(q, expected, group, llm, use_judge) for q, expected, group in GOLDEN]
    rows += [run(q, None, "out of scope", llm, use_judge) for q in OUT_OF_SCOPE]
    df = pd.DataFrame(rows).astype({"rank": "Int64"})      # Int64: a whole number or <NA> (no right source found)
    hidden = ["cost", "judge cost", "claims", "bad cites", "answer"] + ([] if llm else [
        "system", "refused", "first word ms", "lang ok", "cites ok", "wrong claims", "answers"])
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(df.drop(columns=hidden).to_string(index=False))

    answerable, out = df[df["group"] != "out of scope"], df[df["group"] == "out of scope"]
    # no right source found (<NA>) is a miss: without fillna, pandas would leave it out of the mean
    answerable = answerable.assign(hit1=(answerable["rank"] == 1).fillna(False).astype(bool),
                                   hit3=(answerable["rank"] <= 3).fillna(False).astype(bool))
    hit1, hit3 = answerable["hit1"].mean(), answerable["hit3"].mean()
    print(f"\nSearch, {len(answerable)} answerable questions: hit@1 {hit1:.0%}, hit@3 {hit3:.0%}")
    print((answerable.groupby("group")["hit3"]
           .agg(lambda s: f"{s.sum()}/{len(s)}")).to_string())
    print("\nGrade per kind of question:")
    kind = pd.Series(np.where(df["group"] == "out of scope", "out of scope", "answerable"), name="question")
    print(pd.crosstab(kind, df["grade"]).to_string())

    out_answered = 0
    if llm:
        right_top3 = answerable[answerable["hit3"]]
        out_answered = int((~out["refused"]).sum())
        print(f"\nRefused (by the grader or the LLM) although a right source was in the top 3: {int(right_top3['refused'].sum())}/{len(right_top3)}")
        print(f"Out of scope but answered by the LLM: {out_answered}/{len(out)}")
        print("First word on the server (the brief's budget: p95 < 1500 ms):")
        for system, ms in df.groupby("system")["first word ms"]:
            print(f"  {system:20} n {len(ms):2}   p50 {ms.median():5.0f} ms   p95 {np.percentile(ms, 95):5.0f} ms")
        answered = df[df["lang ok"].notna()]
        print(f"\nAnswers to answerable questions ({len(answered)}, refusals not counted):")
        print(f"  in the language of the question: {int(answered['lang ok'].sum())}/{len(answered)}")
        for _, row in answered[~answered["lang ok"].astype(bool)].iterrows():
            print(f"    {row['question']}\n      answer: {row['answer'][:150]}")
        print(f"  citations all copy a source:     {int(answered['cites ok'].sum())}/{len(answered)}")
        for _, row in answered[~answered["cites ok"].astype(bool)].iterrows():
            print(f"    {row['question']}\n      not a source label: {' | '.join(row['bad cites'])}")
        if use_judge:
            judged = answered[answered["wrong claims"].notna()]
            print(f"  judge ({JUDGE_LLM}), no wrong claim:  {int((judged['wrong claims'] == 0).sum())}/{len(judged)}"
                  f"   answers the question: {int(judged['answers'].sum())}/{len(judged)}")
            for _, row in judged[judged["wrong claims"] > 0].iterrows():
                print(f"    {row['question']}")
                for claim in row["claims"]:
                    print(f"      - {claim}")
        print(f"Cost: ${df['cost'].sum():.3f} for {len(df)} questions, judge ${df['judge cost'].sum():.3f}")

    outside, fiod_found = leaks()
    print(f"\nLeak test: {outside} chunks outside the role in {len(ROLES) * len(LEAK_TESTS)} searches"
          f" (fiod finds FIOD documents: {'yes' if fiod_found else 'NO, the test proves nothing'})")

    checks = [
        ("leaks", outside <= GATES["leaks"] and fiod_found, f"{outside} chunks outside the role"),
        ("min_hit_at_3", hit3 >= GATES["min_hit_at_3"], f"hit@3 {hit3:.0%}, gate {GATES['min_hit_at_3']:.0%}"),
        ("out_of_scope_relevant", (out["grade"] == "Relevant").sum() <= GATES["out_of_scope_relevant"],
         f"{int((out['grade'] == 'Relevant').sum())} out-of-scope questions graded Relevant"),
    ]
    if llm:
        checks.append(("out_of_scope_answered", out_answered <= GATES["out_of_scope_answered"],
                       f"{out_answered} out-of-scope questions answered"))
    print("\nGates:")
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:24} {detail}")
    sys.exit(0 if all(ok for _, ok, _ in checks) else 1)


if __name__ == "__main__":
    main()
