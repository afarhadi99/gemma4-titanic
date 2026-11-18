import pytest

from titanic.llm import chat_prompt, pick_label_tokens


def enc(table):
    return lambda w: table.get(w, [1, 2])


def test_prefers_plain_words():
    words, ids = pick_label_tokens(enc({"no": [5], "yes": [6], " no": [7], " yes": [8]}))
    assert words == ("no", "yes") and ids == [5, 6]


def test_falls_back_to_space_then_capitalised():
    assert pick_label_tokens(enc({"no": [5, 9], "yes": [6], " no": [7], " yes": [8]}))[0] == (" no", " yes")
    words, ids = pick_label_tokens(enc({"No": [3], "Yes": [4]}))
    assert words == ("No", "Yes") and ids == [3, 4]


def test_rejects_colliding_tokens_and_raises_when_none():
    assert pick_label_tokens(enc({"no": [5], "yes": [5], " no": [7], " yes": [8]}))[1] == [7, 8]
    with pytest.raises(ValueError):
        pick_label_tokens(enc({}))


class _Tok:
    def __init__(self, accepts):
        self.accepts = accepts

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
        if kw and not self.accepts:
            raise TypeError("unexpected kwarg")
        return f"prompt think={kw.get('enable_thinking')}"


def test_chat_prompt_passes_enable_thinking_and_falls_back():
    assert chat_prompt(_Tok(True), []) == "prompt think=False"
    assert chat_prompt(_Tok(False), []) == "prompt think=None"
