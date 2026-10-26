"""Calls to Claude, the prompts, and how sources are shown to the model (NB5)."""
import queue
import threading
import time

import anthropic

from app.config import EFFORT, HEDGE_AFTER_MS, LLMS, THINKING

_client = None


def client():
    """Created on first use, so the app can start (and /health answer) before a key is set."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic()        # reads ANTHROPIC_API_KEY from the environment (.env), never from code
    return _client


def _open(llm_key, request):
    if llm_key == "sonnet":      # if the safety filters refuse, the API retries on another model (beta)
        return client().beta.messages.stream(**request, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    return client().messages.stream(**request)


def _run(llm_key, request, call, events, stop):
    """One streamed call in its own thread. Every piece goes on the queue, marked with the call that sent it."""
    try:
        with _open(llm_key, request) as stream:
            for piece in stream.text_stream:
                if stop.is_set():
                    return                      # the other call won: leaving the with-block closes this one
                events.put((call, "text", piece))
            events.put((call, "done", stream.get_final_message()))
    except Exception as error:
        events.put((call, "error", error))


def ask_llm(llm_key, system, user, max_tokens=4000, on_text=None):
    """One streamed call, or two for a hedged model (see HEDGE_AFTER_MS). Returns the text and the numbers of the
    call (time to first token, cost, hedged).

    on_text: called with every piece of text as it arrives, so the page can show it at once.
    """
    llm = LLMS[llm_key]
    request = dict(model=llm["name"], max_tokens=max_tokens, system=system,
                   messages=[{"role": "user", "content": user}],
                   output_config={"effort": EFFORT[llm_key]})
    if THINKING.get(llm_key):
        request["thinking"] = THINKING[llm_key]
    events, stops, errors = queue.Queue(), {1: threading.Event(), 2: threading.Event()}, []
    start, ttft, pieces, winner, message = time.perf_counter(), None, [], None, None
    threading.Thread(target=_run, args=(llm_key, request, 1, events, stops[1]), daemon=True).start()
    calls, hedge_after = 1, HEDGE_AFTER_MS.get(llm_key)
    while message is None:
        wait = None
        if hedge_after and calls == 1 and winner is None:
            wait = max(0.0, hedge_after / 1000 - (time.perf_counter() - start))
        try:
            call, kind, payload = events.get(timeout=wait)
        except queue.Empty:                     # no first word yet: the same call once more, first to answer wins
            calls = 2
            threading.Thread(target=_run, args=(llm_key, request, 2, events, stops[2]), daemon=True).start()
            continue
        if kind == "error":
            if call == winner:
                raise payload
            errors.append(payload)
            if winner is None and len(errors) == calls:
                raise errors[0]
            continue
        if winner is None:
            winner = call
            stops[3 - call].set()
        if call != winner:
            continue
        if kind == "text":
            if ttft is None:
                ttft = (time.perf_counter() - start) * 1000
            pieces.append(payload)
            if on_text:
                on_text(payload)
        else:
            message = payload
    total = (time.perf_counter() - start) * 1000
    text = "".join(pieces)
    if message.stop_reason == "refusal":
        text = "NOT_IN_SOURCES"
    usage = message.usage
    inputs = usage.input_tokens * calls       # the other call of a hedge is billed for its input too
    cost = (inputs * llm["price_in"] + usage.output_tokens * llm["price_out"]) / 1e6
    return text, {"ttft_ms": round(ttft or total), "total_ms": round(total), "cost": cost, "hedged": calls == 2}


SYSTEM_PROMPT = """You answer questions about Dutch income tax for employees of the Dutch Tax Administration.

Rules:
1. Use only the sources between the <sources> tags. Do not use your own knowledge, even if you are sure.
2. After every claim, cite its source in parentheses, copying the source's cite attribute exactly, for example (Wet IB 2001, art. 3.114 lid 1-2).
3. If the sources do not contain the answer, or the question asks about a tax year that the law sources do not cover, reply only with: NOT_IN_SOURCES
4. Answer in {language}, in at most {sentences} sentences.
5. Never calculate tax amounts yourself. If the question needs a calculation, quote the rule and the numbers from the source and say that the tax calculator does the calculation."""

LANGUAGE_NAMES = {"nl": "Dutch", "en": "English"}


def system_prompt(language, sentences):
    return SYSTEM_PROMPT.replace("{language}", LANGUAGE_NAMES[language]).replace("{sentences}", str(sentences))


REWRITE_PROMPT = """Rewrite the user's question as one short search query in Dutch, using the legal terms of the Dutch income tax act (Wet IB 2001), for example "eigenwoningschuld" instead of "hypotheekrente" and "werkruimte" instead of "thuiskantoor".
Reply with the query only."""

REFUSAL_MESSAGE = "Dit staat niet in de beschikbare bronnen. / This is not in the available sources."
STATUS_MESSAGE = "Ik zoek dit grondig uit... / Looking into this carefully..."   # shown while System 2 thinks
LLM_ERROR_MESSAGE = "Claude is nu niet bereikbaar, probeer het opnieuw. / Claude is unavailable right now, please try again."


def is_refusal(text):
    return "NOT_IN_SOURCES" in text


def source_label(row):
    if row["doc_type"] == "legislation":
        return f"{row['law']}, art. {row['article']} lid {row['paragraph']}"
    if row["doc_type"] == "ruling":
        return f"{row['ecli']}, {row['law']}"
    return row["law"]


def format_sources(rows):
    blocks = []
    for i, (_, row) in enumerate(rows.iterrows(), start=1):
        if row["doc_type"] == "legislation":
            date = f"tax year {row['year']}"
        elif row["doc_type"] == "ruling":
            date = f"decided {row['valid_from']}"
        else:
            date = f"document from {row['year']}"
        blocks.append(f'<source id="{i}" cite="{source_label(row)}" date="{date}">\n{row["text"]}\n</source>')
    return "\n\n".join(blocks)
