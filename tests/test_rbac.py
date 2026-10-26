"""NB6: no role ever receives a chunk above its level."""
import pytest

from app.config import ROLES
from app.golden import LEAK_TESTS
from app.search import retrieve_scored, search_qdrant


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("question", LEAK_TESTS)
def test_no_chunk_above_the_role(role, question):
    candidates = search_qdrant(question, role)                # every candidate, before the reranker
    assert set(candidates["classification"]) <= set(ROLES[role])
    rows, _, _ = retrieve_scored(question, question, role)    # what the LLM reads
    assert set(rows["classification"]) <= set(ROLES[role])


def test_fiod_finds_fiod_documents():
    """Without this, the leak test could pass only because nobody finds FIOD documents at all."""
    found = search_qdrant("Wordt Alpha BV onderzocht door de FIOD?", "fiod")
    assert "fiod" in set(found["classification"])
