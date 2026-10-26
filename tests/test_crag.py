"""The CRAG decisions of NB7, with a fake search and a fake LLM: every path is tested, nothing is paid."""
import pandas as pd
import pytest

import app.graph as graph
from app.config import TAU_HIGH, TAU_LOW
from app.llm import REFUSAL_MESSAGE

SOURCE = pd.DataFrame([{"doc_type": "legislation", "law": "Wet IB 2001", "article": "3.16", "paragraph": "13",
                        "year": "2026", "parent_id": "art3.16", "classification": "public", "text": "..."}])
RELEVANT, AMBIGUOUS, IRRELEVANT = TAU_HIGH + 1, (TAU_LOW + TAU_HIGH) / 2, TAU_LOW - 1


def run(monkeypatch, score, complex_=False, answer="Ja (Wet IB 2001, art. 3.16 lid 13)."):
    """Asks one question with the given grader score; returns the result and which LLMs were called."""
    calls = []

    def fake_llm(llm_key, system, user, max_tokens=4000, on_text=None):
        calls.append(llm_key)
        return answer, {"ttft_ms": 1, "total_ms": 1, "cost": 0.0}

    monkeypatch.setattr(graph, "route_question", lambda question: ("nl", complex_, 0.9 if complex_ else 0.1))
    monkeypatch.setattr(graph, "retrieve_scored", lambda search, question, role: (SOURCE, score, score is None))
    monkeypatch.setattr(graph, "ask_llm", fake_llm)
    monkeypatch.setattr(graph, "warm_connection", lambda: None)
    return graph.ask("vraag", "helpdesk"), calls


def test_relevant_simple_goes_to_system_1(monkeypatch):
    result, calls = run(monkeypatch, RELEVANT)
    assert result["grade"] == "Relevant" and calls == ["haiku"] and result["status_ms"] is None


def test_relevant_complex_goes_to_system_2_with_status(monkeypatch):
    result, calls = run(monkeypatch, RELEVANT, complex_=True)
    assert calls == ["sonnet"] and result["status_ms"] is not None


def test_ambiguous_goes_to_system_2_without_retry(monkeypatch):
    """MAX_RETRIES = 0: one rewrite costs ~913 ms and does not fit the 1.5 s budget."""
    result, calls = run(monkeypatch, AMBIGUOUS)
    assert result["grade"] == "Ambiguous" and calls == ["sonnet"] and result["status_ms"] is not None


def test_irrelevant_is_refused_without_llm(monkeypatch):
    result, calls = run(monkeypatch, IRRELEVANT)
    assert result["grade"] == "Irrelevant" and calls == [] and result["answer"] == REFUSAL_MESSAGE


def test_ecli_match_is_relevant_without_score(monkeypatch):
    result, calls = run(monkeypatch, None)
    assert result["grade"] == "Relevant" and calls == ["haiku"]


def test_model_refusal_shows_the_fixed_message(monkeypatch):
    result, _ = run(monkeypatch, RELEVANT, answer="NOT_IN_SOURCES")
    assert result["answer"] == REFUSAL_MESSAGE


@pytest.mark.parametrize("score, grade", [(TAU_HIGH + 0.001, "Relevant"), (TAU_HIGH, "Ambiguous"),
                                          (TAU_LOW, "Ambiguous"), (TAU_LOW - 0.001, "Irrelevant")])
def test_thresholds(score, grade):
    assert graph.zone(score) == grade


def test_stream_never_shows_the_refusal_code():
    written = []
    stream = graph.AnswerStream(written.append)
    for piece in ["NOT_", "IN_", "SOURCES"]:
        stream(piece)
    assert written == []
    stream = graph.AnswerStream(written.append)
    stream("Ja, ")
    assert written == [{"type": "token", "text": "Ja, "}]
