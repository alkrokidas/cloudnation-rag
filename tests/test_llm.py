"""The hedged Haiku call: a late first word sends the same call once more, and the first to answer wins."""
import time
from types import SimpleNamespace

import pytest

from app import llm


class FakeStream:
    def __init__(self, call, delay, fail):
        self.call, self.delay, self.fail = call, delay, fail

    def __enter__(self):
        time.sleep(self.delay)                  # the time to the first word
        if self.fail:
            raise RuntimeError(f"call {self.call} failed")
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        yield f"call {self.call} "
        yield "answers"

    def get_final_message(self):
        return SimpleNamespace(stop_reason="end_turn", usage=SimpleNamespace(input_tokens=1000, output_tokens=10))


class FakeClient:
    """Each new call takes the next delay (seconds to its first word); fail=True makes every call fail."""
    def __init__(self, delays, fail=False):
        self.delays, self.fail, self.calls = list(delays), fail, 0
        self.messages = SimpleNamespace(stream=self.stream)
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self.stream))

    def stream(self, **request):
        self.calls += 1
        return FakeStream(self.calls, self.delays[self.calls - 1], self.fail)


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setitem(llm.HEDGE_AFTER_MS, "haiku", 50)
    def install(delays, fail=False):
        client = FakeClient(delays, fail)
        monkeypatch.setattr(llm, "client", lambda: client)
        return client
    return install


def test_a_fast_first_word_sends_one_call(fake):
    client = fake([0.0])
    text, stats = llm.ask_llm("haiku", "system", "user")
    assert text == "call 1 answers" and not stats["hedged"] and client.calls == 1


def test_a_late_first_word_sends_a_second_call_and_the_first_to_answer_wins(fake):
    client = fake([0.5, 0.0])
    shown = []
    text, stats = llm.ask_llm("haiku", "system", "user", on_text=shown.append)
    assert text == "call 2 answers" and "".join(shown) == text      # the page sees only the winner
    assert stats["hedged"] and client.calls == 2 and stats["ttft_ms"] < 400
    assert stats["cost"] == pytest.approx((2 * 1000 * 0.10 + 10 * 0.50) / 1e6)   # both inputs are billed


def test_the_first_call_still_wins_if_it_answers_before_the_second(fake):
    client = fake([0.1, 0.5])
    text, stats = llm.ask_llm("haiku", "system", "user")
    assert text == "call 1 answers" and stats["hedged"] and client.calls == 2


def test_both_calls_failing_raises(fake):
    fake([0.1, 0.0], fail=True)
    with pytest.raises(RuntimeError):
        llm.ask_llm("haiku", "system", "user")


def test_sonnet_is_not_hedged(fake):
    client = fake([0.2])
    text, stats = llm.ask_llm("sonnet", "system", "user")
    assert text == "call 1 answers" and not stats["hedged"] and client.calls == 1
