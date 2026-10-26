"""Search rules the notebooks showed are needed."""
import pytest

from app.config import MAX_CHUNKS_PER_DOC
from app.golden import GOLDEN
from app.search import retrieve_scored


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
