"""The API: POST /ask runs the CRAG graph for the logged-in user's role. Start: uvicorn app.main:app"""
import json
import math
import time
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app import ingest
from app.config import COLLECTION, DEMO_USERS, SYSTEM1_MODEL
from app.graph import ask, crag, initial_state
from app.llm import LLM_ERROR_MESSAGE, STATUS_MESSAGE, ask_llm, source_label
from app.router import route_question
from app.search import qdrant, retrieve_scored

WARM_UP_QUESTION = "Welke kosten van een werkruimte in de woning zijn aftrekbaar?"   # not the demo question: no cache hit


def warm_up(runs=2):
    """The first search in a new process is slow (PyTorch sets up memory and kernels), so we run it before the
    first user does. No LLM call. The log shows each run: the last one is the real speed of this machine."""
    for n in range(1, runs + 1):
        start = time.perf_counter()
        route_question(WARM_UP_QUESTION)
        retrieve_scored(WARM_UP_QUESTION, WARM_UP_QUESTION, "helpdesk")
        print(f"Warm-up search {n}/{runs}: {(time.perf_counter() - start) * 1000:.0f} ms", flush=True)


def warm_up_llm():
    """The first Claude call of a process is slow (26/10: 5226 ms against 1200-1570 ms for the next ones), so one
    tiny call runs before the first user does. Costs about $0.00001."""
    start = time.perf_counter()
    try:
        ask_llm(SYSTEM1_MODEL, "Reply with: OK", "OK", max_tokens=5)
    except Exception as error:      # no key yet: the app still starts, and the first question reports the error
        print(f"Warm-up Claude: skipped ({type(error).__name__})", flush=True)
        return
    print(f"Warm-up Claude: {(time.perf_counter() - start) * 1000:.0f} ms", flush=True)


@asynccontextmanager
async def lifespan(app):
    if not qdrant.collection_exists(COLLECTION):   # without Docker there is no ingest step: build it here
        ingest.build()
    warm_up()
    warm_up_llm()
    yield


app = FastAPI(title="Belastingdienst RAG prototype", lifespan=lifespan)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def role_from_token(authorization):
    """Authorization: Bearer <token> -> role. In production the token comes from the SSO login."""
    token = (authorization or "").removeprefix("Bearer ").strip()
    if token not in DEMO_USERS:
        raise HTTPException(status_code=401, detail="Unknown user")
    return DEMO_USERS[token]


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "chunks": qdrant.count(COLLECTION).count}


def llm_error(error):
    """Claude timed out, is overloaded (529), rate-limited (429) or rejected the key: say so instead of a 500."""
    print(f"Claude API error: {error!r}", flush=True)
    return f"{LLM_ERROR_MESSAGE} ({type(error).__name__})"


def response(result, role):
    """What both endpoints return about a finished question."""
    rows, score = result["rows"], result["score"]
    sources = [] if rows is None or result["grade"] == "Irrelevant" else [source_label(r) for _, r in rows.iterrows()]
    return {
        "role": role,
        "answer": result["answer"],
        "system": result["system"],
        "grade": result["grade"],
        "score": score if score is None or math.isfinite(score) else None,   # -inf (nothing found) is not valid JSON
        "language": result["language"],
        "complex": result["complex"],
        "status_message": STATUS_MESSAGE if result["status_ms"] is not None else None,
        "status_ms": result["status_ms"],
        "ttft_ms": round(result["ttft_ms"]),
        "cost_usd": round(result["cost"], 5),
        "trace": [{"node": name, "ms": ms} for name, ms in result["trace"]],
        "sources": sources,
    }


@app.post("/ask")
def ask_endpoint(body: Question, authorization: str | None = Header(default=None)):
    role = role_from_token(authorization)
    try:
        result = ask(body.question, role)
    except anthropic.AnthropicError as error:
        raise HTTPException(status_code=503, detail=llm_error(error)) from error
    return response(result, role)


def sse(event):
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/ask/stream")
def ask_stream(body: Question, authorization: str | None = Header(default=None)):
    """The same question as POST /ask, answered as server-sent events, so the page shows the first word at the
    LLM's time to first token instead of after the whole answer:
    {"type": "status"} for System 2, {"type": "token"} per piece of the answer, then {"type": "done"} with
    everything /ask returns (or {"type": "error"})."""
    role = role_from_token(authorization)          # before the stream starts, so a wrong token is still a 401

    def events():
        result = None
        try:
            for mode, chunk in crag.stream(initial_state(body.question, role), stream_mode=["custom", "values"]):
                if mode == "custom":
                    yield sse(chunk)
                else:
                    result = chunk                 # the state after each step; the last one is the final state
        except anthropic.AnthropicError as error:
            yield sse({"type": "error", "detail": llm_error(error)})
            return
        yield sse({"type": "done", **response(result, role)})

    # no-cache and X-Accel-Buffering: proxies (like the Codespaces port forwarding) must pass each event on at once
    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
