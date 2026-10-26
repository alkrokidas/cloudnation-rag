"""Search rules the notebooks showed are needed."""
import pytest

from app.config import MAX_CHUNKS_PER_DOC
from app.golden import GOLDEN
from app.search import retrieve_scored, with_legal_terms


@pytest.mark.parametrize("question", [q for q, _, group in GOLDEN if group != "ecli"])
def test_at_most_two_chunks_per_document(question):
    """NB5-NB7: 4 of 5 chunks often came from one ruling and pushed out the law article."""
    rows, _, _ = retrieve_scored(question, question, "helpdesk")
    assert rows["parent_id"].value_counts().max() <= MAX_CHUNKS_PER_DOC


def test_ecli_is_an_exact_lookup():
    """NB3: a case number is a primary-key lookup, not a similarity search."""
    rows, score, exact = retrieve_scored("ECLI:NL:HR:2014:2872", "ECLI:NL:HR:2014:2872", "helpdesk")
    assert exact and score is None
    assert set(rows["ecli"]) == {"ECLI:NL:HR:2014:2872"}


def test_legal_terms_are_added_not_swapped():
    assert (with_legal_terms("Mag ik de kosten van mijn werkkamer thuis aftrekken?")
            == "Mag ik de kosten van mijn werkkamer thuis aftrekken? werkruimte woning")
    assert with_legal_terms("Is mijn werkruimte in de woning aftrekbaar?") == "Is mijn werkruimte in de woning aftrekbaar?"


def test_everyday_words_find_the_paragraph_of_the_law():
    """26/10: "werkkamer thuis" left art. 3.16 lid 1 at search rank 14, outside the 10 the reranker reads, and
    Haiku answered from lid 13 (a different case), once with the meaning reversed."""
    question = "Mag ik de kosten van mijn werkkamer thuis aftrekken?"
    rows, _, _ = retrieve_scored(question, question, "helpdesk")
    assert "wetib2001-2026-art3.16-lid1" in set(rows["chunk_id"])
