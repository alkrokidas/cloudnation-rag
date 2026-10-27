"""Calls to Claude, the prompts, and how sources are shown to the model (NB5)."""
import time

import anthropic

from app.config import EFFORT, LLMS, THINKING

_client = None


def client():
    """Created on first use, so the app can start (and /health answer) before a key is set."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic()        # reads ANTHROPIC_API_KEY from the environment (.env), never from code
    return _client


def ask_llm(llm_key, system, user, max_tokens=4000, on_text=None):
    """One streamed call. Returns the text and the numbers of the call (time to first token, cost).

    on_text: called with every piece of text as it arrives, so the page can show it at once.
    """
    llm = LLMS[llm_key]
    request = dict(model=llm["name"], max_tokens=max_tokens, system=system,
                   messages=[{"role": "user", "content": user}],
                   output_config={"effort": EFFORT[llm_key]})
    if THINKING.get(llm_key):
        request["thinking"] = THINKING[llm_key]
    if llm_key == "sonnet":      # if the safety filters refuse, the API retries on another model (beta)
        call = client().beta.messages.stream(**request, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    else:
        call = client().messages.stream(**request)
    start, ttft, pieces = time.perf_counter(), None, []
    with call as stream:
        for piece in stream.text_stream:
            if ttft is None:
                ttft = (time.perf_counter() - start) * 1000
            pieces.append(piece)
            if on_text:
                on_text(piece)
        message = stream.get_final_message()
    total = (time.perf_counter() - start) * 1000
    text = "".join(pieces)
    if message.stop_reason == "refusal":
        text = "NOT_IN_SOURCES"
    usage = message.usage
    cost = (usage.input_tokens * llm["price_in"] + usage.output_tokens * llm["price_out"]) / 1e6
    return text, {"ttft_ms": round(ttft or total), "total_ms": round(total), "cost": cost}


# Rule 5 also said "say that the tax calculator does the calculation" until 26/10: the judge flagged that sentence in
# 5 of 19 answers (no source mentions a calculator), often where no calculation was asked.
# Rules 6 and 7, from the judge on the golden set (26-27/10): 3 of 21 answers gave the rechtbank's words to the Hof
# (a Hof ruling quotes the rechtbank before its own judgement, and the cite label names only the Hof), and a home
# office question merged art. 3.16 lid 1 sub a and b in both runs (English, then Dutch), dropping a condition.
SYSTEM_PROMPT = """You answer questions about Dutch income tax for employees of the Dutch Tax Administration.

Rules:
1. Use only the sources between the <sources> tags. Do not use your own knowledge, even if you are sure.
2. After every claim, cite its source in parentheses, copying the source's cite attribute exactly, for example (Wet IB 2001, art. 3.114 lid 1-2).
3. If the sources do not contain the answer, or the question asks about a tax year that the law sources do not cover, reply only with: NOT_IN_SOURCES
4. Answer in {language}, in at most {sentences} sentences.
5. Never calculate tax amounts yourself. If the question needs a calculation, quote the rule and the numbers from the source instead.
6. A ruling often quotes the lower court (rechtbank) or a party before its own judgement. Say that a court decided something only if the text says that court decided it.
7. If a source lists separate cases with their own conditions (for example a and b), keep each case with its own conditions. Never merge them into one rule."""

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
