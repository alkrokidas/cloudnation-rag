"""The answer language is decided in code (NB5): small words, then tax words; a tie is Dutch."""
import pytest

from app.router import EN_WORDS, NL_WORDS, detect_language, route_question


@pytest.mark.parametrize("question, language", [
    ("Mag ik de kosten van mijn werkkamer thuis aftrekken?", "nl"),
    ("Can I deduct the costs of a home office?", "en"),
    ("airbnb taxes", "en"),                              # keywords only: the tax words decide (23/10)
    ("hypotheekrente aftrek", "nl"),
    ("Mag ik mijn home office aftrekken?", "nl"),        # English words in a Dutch sentence
    ("ECLI:NL:GHDHA:2020:887", "nl"),                    # a tie: Dutch, the language of the sources
])
def test_language(question, language):
    assert detect_language(question) == language


def test_no_word_in_both_lists():
    assert not NL_WORDS & EN_WORDS


def test_bare_case_number_is_simple():
    """26/10: the classifier called a bare ECLI complex and sent it to Sonnet (4.3 s instead of ~0.5 s)."""
    assert route_question("ECLI:NL:HR:2014:2872")[1] is False
