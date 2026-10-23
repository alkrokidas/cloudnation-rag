"""The API: POST /ask runs the CRAG graph for the logged-in user's role. Start: uvicorn app.main:app"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import ingest
from app.config import COLLECTION, DEMO_USERS
from app.graph import ask
from app.llm import STATUS_MESSAGE, source_label
from app.search import qdrant


@asynccontextmanager
async def lifespan(app):
    if not qdrant.collection_exists(COLLECTION):   # without Docker there is no ingest step: build it here
        ingest.build()
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


@app.post("/ask")
def ask_endpoint(body: Question, authorization: str | None = Header(default=None)):
    role = role_from_token(authorization)
    result = ask(body.question, role)
    rows = result["rows"]
    sources = [] if rows is None or result["grade"] == "Irrelevant" else [source_label(r) for _, r in rows.iterrows()]
    return {
        "role": role,
        "answer": result["answer"],
        "system": result["system"],
        "grade": result["grade"],
        "score": result["score"],
        "language": result["language"],
        "complex": result["complex"],
        "status_message": STATUS_MESSAGE if result["status_ms"] is not None else None,
        "status_ms": result["status_ms"],
        "ttft_ms": round(result["ttft_ms"]),
        "cost_usd": round(result["cost"], 5),
        "trace": [{"node": name, "ms": ms} for name, ms in result["trace"]],
        "sources": sources,
    }
