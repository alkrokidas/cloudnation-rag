"""Checks on the answers themselves, used by python -m app.evaluate (Module 5).

hit@k says whether the right source was found, not whether the answer is right: on 26/10 Haiku answered the demo
question once with the law reversed, and no gate saw it. Three checks per answer:
  language   - the answer is in the language the router chose (free)
  citations  - it cites at least one source, and every citation is one of its sources, like a foreign key (free, NB5)
  judge      - a stronger model reads sources, question and answer and lists the claims the sources do not support
"""
import json
import re

from app.config import LLMS
from app.llm import client, format_sources, source_label
from app.router import detect_language

JUDGE_LLM = "sonnet"      # stronger than System 1. System 2 is Sonnet too, so there it checks its own model;
                          # Opus would be independent at about twice the cost per run.

JUDGE_PROMPT = """You check the answers of an assistant for the Dutch Tax Administration. You get the sources the assistant read, the question and its answer.

For every claim in the answer, check that the source it cites says the same thing. Watch negations and exceptions ("niet ... tenzij", "met uitzondering van", "unless"): an answer that turns an exception into the rule, or the rule into an exception, is wrong.
Judge only what the answer says, not its style and not what it leaves out.

wrong_claims: every claim the sources do not support or contradict, each with one short reason. Empty if all claims are supported.
answers_the_question: true if the answer gives the user an answer to their question, false if it only says the sources are incomplete."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"wrong_claims": {"type": "array", "items": {"type": "string"}},
                   "answers_the_question": {"type": "boolean"}},
    "required": ["wrong_claims", "answers_the_question"],
    "additionalProperties": False,
}

CITATION = re.compile(r"\(([^()]*(?:art\.|ECLI:)[^()]*)\)")   # (Wet IB 2001, art. 3.16 lid 1) or (ECLI:..., court)


def language_ok(answer, language):
    return detect_language(answer) == language


def bad_citations(answer, rows):
    """The citations that copy no label of a source the assistant was given; ["no citation"] if there is none."""
    labels = {source_label(row) for _, row in rows.iterrows()}
    cited = [part.strip() for group in CITATION.findall(answer) for part in group.split(";")]
    return [c for c in cited if c not in labels] if cited else ["no citation"]


def citations_ok(answer, rows):
    """At least one citation, and each one copies the label of a source the assistant was given."""
    return not bad_citations(answer, rows)


def judge(question, rows, answer):
    """Returns ({"wrong_claims": [...], "answers_the_question": bool} or None if the judge declined, cost in $)."""
    llm = LLMS[JUDGE_LLM]
    user = f"<sources>\n{format_sources(rows)}\n</sources>\n\nQuestion: {question}\n\nAnswer: {answer}"
    response = client().beta.messages.create(
        model=llm["name"], max_tokens=16000, system=JUDGE_PROMPT, messages=[{"role": "user", "content": user}],
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    usage = response.usage
    cost = (usage.input_tokens * llm["price_in"] + usage.output_tokens * llm["price_out"]) / 1e6
    if response.stop_reason == "refusal":
        return None, cost
    text = next(block.text for block in response.content if block.type == "text")
    return json.loads(text), cost
