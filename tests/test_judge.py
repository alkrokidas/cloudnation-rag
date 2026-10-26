"""The checks on the answers themselves (app/judge.py)."""
import json
from types import SimpleNamespace

import pandas as pd

from app import judge

ROWS = pd.DataFrame([
    {"doc_type": "legislation", "law": "Wet IB 2001", "article": "3.16", "paragraph": "1", "year": "2026",
     "valid_from": "", "ecli": "", "text": "Bij het bepalen van de winst komen mede niet in aftrek ..."},
    {"doc_type": "ruling", "law": "Gerechtshof Arnhem-Leeuwarden", "article": "", "paragraph": "", "year": "2026",
     "valid_from": "2026-02-03", "ecli": "ECLI:NL:GHARL:2026:698", "text": "Het hof oordeelt ..."},
])


def test_citations_must_copy_a_source():
    assert judge.citations_ok("Dat mag, onder voorwaarden (Wet IB 2001, art. 3.16 lid 1).", ROWS)
    assert judge.citations_ok("Zo oordeelde het hof (ECLI:NL:GHARL:2026:698, Gerechtshof Arnhem-Leeuwarden; "
                              "Wet IB 2001, art. 3.16 lid 1).", ROWS)
    assert not judge.citations_ok("Dat mag (Wet IB 2001, art. 3.16 lid 13).", ROWS)      # not one of its sources
    assert not judge.citations_ok("Dat mag.", ROWS)                                        # no citation at all
    assert judge.bad_citations("Dat mag (Wet IB 2001, art. 3.16 lid 13).", ROWS) == ["Wet IB 2001, art. 3.16 lid 13"]
    assert judge.bad_citations("Dat mag.", ROWS) == ["no citation"]


def test_language_follows_the_router():
    assert judge.language_ok("Of de kosten aftrekbaar zijn, hangt af van de werkruimte.", "nl")
    assert not judge.language_ok("The sources do not directly answer whether you can deduct the costs.", "nl")


def test_judge_reads_the_structured_verdict(monkeypatch):
    verdict = {"wrong_claims": ["Says the costs are not deductible; lid 13 makes them an exception."],
               "answers_the_question": True}
    seen = {}

    def create(**request):
        seen.update(request)
        return SimpleNamespace(stop_reason="end_turn", usage=SimpleNamespace(input_tokens=2000, output_tokens=100),
                               content=[SimpleNamespace(type="text", text=json.dumps(verdict))])

    fake = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    monkeypatch.setattr(judge, "client", lambda: fake)
    result, cost = judge.judge("Mag ik ...?", ROWS, "Nee (Wet IB 2001, art. 3.16 lid 1).")
    assert result == verdict
    assert seen["output_config"]["format"]["schema"] == judge.JUDGE_SCHEMA
    assert "<sources>" in seen["messages"][0]["content"] and "Answer: Nee" in seen["messages"][0]["content"]
    assert cost > 0
