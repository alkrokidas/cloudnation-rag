"""The API: the role comes from the login token, never from the request body (NB6)."""
import json

import pytest
from fastapi.testclient import TestClient

import app.graph as graph
from app.main import app as api


def fake_llm(llm_key, system, user, max_tokens=4000, on_text=None):
    text = "Antwoord uit de bronnen."
    if on_text:
        on_text(text)
    return text, {"ttft_ms": 1, "total_ms": 1, "cost": 0.0}


@pytest.fixture(scope="module")
def client():
    with pytest.MonkeyPatch.context() as patch:   # no Claude calls in these tests
        patch.setattr(graph, "ask_llm", fake_llm)
        patch.setattr(graph, "warm_connection", lambda: None)
        with TestClient(api) as test_client:
            yield test_client


def test_unknown_token_is_refused(client):
    assert client.post("/ask", json={"question": "x"}).status_code == 401
    assert client.post("/ask", json={"question": "x"}, headers={"Authorization": "Bearer fiod"}).status_code == 401


def test_empty_question_is_refused(client):
    assert client.post("/ask", json={"question": ""}, headers={"Authorization": "Bearer demo-fiod"}).status_code == 422


def test_role_in_the_body_is_ignored(client):
    r = client.post("/ask", json={"question": "Wordt Alpha BV onderzocht door de FIOD?", "role": "fiod"},
                    headers={"Authorization": "Bearer demo-helpdesk"})
    assert r.status_code == 200
    assert r.json()["role"] == "helpdesk"
    assert not any("FIOD" in source for source in r.json()["sources"])


def test_stream_sends_the_answer_then_done(client):
    with client.stream("POST", "/ask/stream", json={"question": "Mag ik de kosten van mijn werkkamer thuis aftrekken?"},
                       headers={"Authorization": "Bearer demo-helpdesk"}) as r:
        events = [json.loads(e[6:]) for e in "".join(r.iter_text()).split("\n\n") if e.startswith("data: ")]
    assert events[-1]["type"] == "done"
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert tokens == events[-1]["answer"]
