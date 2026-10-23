"""The CRAG state machine with LangGraph (NB7).

route -> retrieve -> grade -> generate   (Relevant, or Ambiguous with no retry left: System 2)
                           -> rewrite -> retrieve ...   (Ambiguous, while MAX_RETRIES and TIME_LIMIT_MS allow)
                           -> refuse     (Irrelevant: fixed message, no LLM)
"""
import time
from typing import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.config import MAX_RETRIES, SYSTEM1_MODEL, TAU_HIGH, TAU_LOW, TIME_LIMIT_MS
from app.llm import (REFUSAL_MESSAGE, REWRITE_PROMPT, STATUS_MESSAGE, ask_llm, format_sources, is_refusal,
                     system_prompt, warm_connection)
from app.router import route_question
from app.search import retrieve_scored


class State(TypedDict):
    question: str        # the user's question (never changes)
    role: str            # from the login, never from the request body (NB6)
    query: str           # what we search for (changes on a rewrite)
    language: str        # nl / en, from the router
    complex: bool        # from the router
    p_complex: float
    rows: object         # the chunks (a DataFrame)
    score: object        # reranker score of the best chunk (None for an ECLI match)
    exact: bool
    grade: str           # Relevant / Ambiguous / Irrelevant
    retries: int
    system: str          # which LLM answered
    answer: str
    started: float
    ttft_ms: float       # when the user saw the first word
    status_ms: object    # System 2: when the user saw the status message
    cost: float
    trace: list          # (node, ms) for every step


def elapsed_ms(state):
    return (time.perf_counter() - state["started"]) * 1000


def zone(score):
    if score is None:            # exact ECLI match: like a primary-key lookup, relevant by definition
        return "Relevant"
    if score > TAU_HIGH:
        return "Relevant"
    if score >= TAU_LOW:
        return "Ambiguous"
    return "Irrelevant"


def route(state):
    warm_connection()         # the TLS handshake to the API runs while we search, not after
    language, complex_, p = route_question(state["question"])
    return {"language": language, "complex": complex_, "p_complex": p}


def retrieve(state):
    rows, score, exact = retrieve_scored(state["query"], state["question"], state["role"])
    if state["rows"] is not None and not exact and state["score"] is not None and score <= state["score"]:
        return {}                # the retry found nothing better: keep the first sources
    return {"rows": rows, "score": score, "exact": exact}


def grade(state):
    return {"grade": "Relevant" if state["exact"] else zone(state["score"])}


def rewrite(state):
    text, stats = ask_llm("haiku", REWRITE_PROMPT, state["question"], max_tokens=100)
    return {"query": text.strip(), "retries": state["retries"] + 1, "cost": state["cost"] + stats["cost"]}


class AnswerStream:
    """Sends the answer to the page piece by piece (POST /ask/stream; with invoke the writes go nowhere).
    Text that could still become NOT_IN_SOURCES is held back, so the page never shows the refusal code."""

    def __init__(self, write):
        self.write, self.text, self.sent = write, "", 0

    def __call__(self, piece):
        self.text += piece
        if "NOT_IN_SOURCES".startswith(self.text.strip()):
            return
        self.write({"type": "token", "text": self.text[self.sent:]})
        self.sent = len(self.text)


def generate(state):
    write = get_stream_writer()
    system2 = state["complex"] or state["grade"] == "Ambiguous"    # complex, or the search stayed uncertain
    llm_key = "sonnet" if system2 else SYSTEM1_MODEL
    status_ms = elapsed_ms(state) if system2 else None             # System 2: the status message goes out now
    if system2:
        write({"type": "status", "text": STATUS_MESSAGE, "ms": round(status_ms)})
    user = f"<sources>\n{format_sources(state['rows'])}\n</sources>\n\nQuestion: {state['question']}"
    call_start = elapsed_ms(state)
    text, stats = ask_llm(llm_key, system_prompt(state["language"], 8 if system2 else 5), user,
                          max_tokens=16000 if system2 else 4000, on_text=AnswerStream(write))
    return {"answer": REFUSAL_MESSAGE if is_refusal(text) else text,
            "system": "System 2 (Sonnet)" if system2 else "System 1 (Haiku)",
            "ttft_ms": call_start + stats["ttft_ms"], "status_ms": status_ms, "cost": state["cost"] + stats["cost"]}


def refuse(state):
    get_stream_writer()({"type": "token", "text": REFUSAL_MESSAGE})
    return {"answer": REFUSAL_MESSAGE, "system": "no LLM", "ttft_ms": elapsed_ms(state)}


def after_grade(state):
    """The CRAG decision."""
    if state["grade"] == "Relevant":
        return "generate"
    if state["grade"] == "Irrelevant":
        return "refuse"
    if state["retries"] < MAX_RETRIES and elapsed_ms(state) < TIME_LIMIT_MS:
        return "rewrite"
    return "generate"            # Ambiguous with no time left -> System 2


def timed(name, node):
    """Wraps a node: measures its ms and appends them to the trace."""
    def wrapper(state):
        start = time.perf_counter()
        update = node(state)
        update["trace"] = state["trace"] + [(name, round((time.perf_counter() - start) * 1000))]
        return update
    return wrapper


builder = StateGraph(State)
for name, node in [("route", route), ("retrieve", retrieve), ("grade", grade),
                   ("rewrite", rewrite), ("generate", generate), ("refuse", refuse)]:
    builder.add_node(name, timed(name, node))
builder.add_edge(START, "route")
builder.add_edge("route", "retrieve")
builder.add_edge("retrieve", "grade")
builder.add_conditional_edges("grade", after_grade, {"generate": "generate", "rewrite": "rewrite", "refuse": "refuse"})
builder.add_edge("rewrite", "retrieve")      # the loop
builder.add_edge("generate", END)
builder.add_edge("refuse", END)
crag = builder.compile()


def initial_state(question, role):
    return {"question": question, "role": role, "query": question, "language": "nl", "complex": False,
            "p_complex": 0.0, "rows": None, "score": None, "exact": False, "grade": "", "retries": 0,
            "system": "", "answer": "", "started": time.perf_counter(), "ttft_ms": 0.0,
            "status_ms": None, "cost": 0.0, "trace": []}


def ask(question, role):
    return crag.invoke(initial_state(question, role))
