"""Every decision of the prototype in one place, with the measurement behind it.

Each value was picked in a notebook (NB1-NB7) from a measurement, not by feel.
The spec quotes the same numbers, so this file and the spec must stay in sync.
"""
import os

# --- Where things live (set by docker-compose.yml; the defaults run without Docker) ---
QDRANT_URL = os.getenv("QDRANT_URL", ":memory:")      # ":memory:" = Qdrant inside the Python process
COLLECTION = "wet_ib"
CHUNKS_CSV = os.getenv("CHUNKS_CSV", "data/chunks_v2.csv")   # law + rulings, output of NB3
MOCK_DOCS_JSON = os.getenv("MOCK_DOCS_JSON", "data/mock_docs.json")   # Dutch FICTIEF internal/FIOD docs (NB6)

# --- Retrieval (NB2-NB4) ---
EMBEDDER = "intfloat/multilingual-e5-small"           # NB2: Dutch and English in one vector space, 384 dims
RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"   # NB4: hit@1 50% -> 75%, same as bge-m3, which had p95 3.4 s
DEPTH = 10              # NB4: candidates the reranker reads; 20 gave the same quality
# Codespace bench (2 vCPU, python -m app.bench, 16 questions), reranker time per question:
#   batch 32: 2575 ms -> one pair at a time: 1501 ms with the same scores (no padding up to the longest pair)
#   512 -> 256 tokens per question + chunk: 1726 -> 1220 ms, grade changed on 0/16, top score moved <= 0.22
#   (TAU_HIGH - TAU_LOW = 1.86). The reranker no longer reads the end of long chunks; the LLM still reads all of it.
#   2 threads on the 2 vCPUs (one physical core): 7497 ms, so PyTorch keeps its default
#   4-core codespace (two physical cores): 2 pairs at once, one core each: 746 -> 668 ms, same scores
#   rejected: DEPTH 5 (1129 ms, grade changed on 2/16), int8 weights (875 ms, grade changed on 3/16),
#   bfloat16 math (4740 ms: the codespace CPU, AMD EPYC 7763, has no bf16 hardware),
#   192 tokens (with bf16: grade changed on 1/16)
RERANK_MAX_TOKENS = 256
RERANK_WORKERS = 2      # pairs scored at the same time, one core each (at most the physical cores); same scores
PREFETCH = 50           # candidates from each of dense and BM25 before fusion
RRF_K = 61              # NB3 used k = 60 with ranks from 1; Qdrant counts ranks from 0 (NB6)
TOP_K = 5               # chunks the LLM reads
MAX_CHUNKS_PER_DOC = 2  # NB5-NB7: 4 of 5 chunks often came from one ruling and pushed out the law article
# Everyday Dutch -> the law's own word, added to the search text (the reranker still reads the user's question).
# 26/10 (T4): "Mag ik de kosten van mijn werkkamer thuis aftrekken?" found art. 3.16 lid 1 at search rank 14, cut
# by DEPTH 10, although the reranker would have put it first (0.476 against -0.995); Haiku then answered from lid 13,
# a different case, once with the meaning reversed. Asked in the law's words, lid 1 ranks 1 (4.957). The same gap as
# NB5 (everyday Dutch vs legal terms). A word is added only after a measured miss; in production the list would
# come from the Belastingdienst's own glossary. Rejected: DEPTH 15 (also brings lid 1 in, +5 pairs = ~+0.35 s on CPU).
LEGAL_TERMS = {"werkkamer": "werkruimte", "thuiskantoor": "werkruimte", "thuis": "woning"}

# --- RBAC (NB6): least privilege, every role sees its own level and below ---
ROLES = {
    "helpdesk": ["public"],
    "inspector": ["public", "internal"],
    "fiod": ["public", "internal", "fiod"],
}
# Prototype stand-in for the Belastingdienst login (SSO). The server maps a token to a role;
# the browser never sends a role, so a user cannot pick a higher one.
DEMO_USERS = {
    "demo-helpdesk": "helpdesk",
    "demo-inspector": "inspector",
    "demo-fiod": "fiod",
}

# --- CRAG grader (NB7): the reranker score of the best chunk, 0 ms extra ---
TAU_LOW = -3.748    # lowest score of an answerable question whose right source was found
TAU_HIGH = -1.893   # highest score of an out-of-scope question

# --- Router (NB7): logistic regression on the query embedding ---
# 0.5, not the Bayes 0.09: in leave-one-out both caught 100% of complex questions, but 0.09 also
# sent 31% of simple ones to Sonnet, and Bayes needs calibrated probabilities that 32 examples cannot check.
ROUTER_TAU = 0.50

# --- Retry loop (NB7 step 7): N_max = floor((1500 - 1018) / 913) = 0 ---
MAX_RETRIES = 0         # one LLM rewrite costs ~913 ms and did not change the grade
TIME_LIMIT_MS = 1200    # second bound: no new search after 1.2 s

# --- LLMs (NB5) ---
SYSTEM1_MODEL = "haiku"     # NB5: ~21x cheaper than Sonnet with a similar time to first token
LLMS = {
    "haiku": {"name": "claude-haiku-5-5", "price_in": 0.10, "price_out": 0.50},    # $ per million tokens
    "sonnet": {"name": "claude-sonnet-5-5", "price_in": 2.00, "price_out": 10.00},
}
THINKING = {"haiku": {"type": "disabled"}}    # System 1 answers without thinking; Sonnet thinks as needed (adaptive)
EFFORT = {"haiku": "low", "sonnet": "medium"}
# Rejected 26/10: opening the API connection in the background during the search. Codespace bench: a call on a
# new connection 174 ms, on a reused one 161 ms, so the handshake costs ~13 ms and the extra call was not worth it.

# --- Release gates (python -m app.evaluate exits 1 if one fails): the numbers a new release must keep ---
GATES = {
    "leaks": 0,                   # chunks outside the role in the leak test (NB6: 0 of 90)
    "min_hit_at_3": 0.80,         # right source in the top 3 (NB4: 83% with one right source per question)
    "out_of_scope_relevant": 0,   # out-of-scope questions graded Relevant (NB7: 0 of 10)
    "out_of_scope_answered": 0,   # out-of-scope questions the LLM answered instead of refusing
}
