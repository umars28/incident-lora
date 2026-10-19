import math

import pytest

from incident_lora.train import answer_span, encode, lr_at


class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True,
                            return_dict=False):
        ids = []
        for m in messages:
            if m["role"] == "assistant":
                ids += [3] + [ord(c) for c in m["content"]] + [2]
            else:
                ids += [1] + [ord(c) for c in m["content"]] + [2]
        if add_generation_prompt:
            ids += [3]
        return ids


RECORD = {
    "id": "x",
    "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "alert"}],
    "target": '{"a":1}',
}


def test_only_answer_tokens_carry_loss():
    enc = encode(FakeTokenizer(), RECORD, max_len=100)
    labelled = [t for t in enc["labels"] if t != -100]
    assert labelled[:-1] == [ord(c) for c in RECORD["target"]]
    assert answer_span(enc["labels"]) == len(labelled)
    assert len(enc["labels"]) == len(enc["input_ids"])


def test_overlong_examples_are_rejected_not_truncated():
    with pytest.raises(ValueError):
        encode(FakeTokenizer(), RECORD, max_len=10)


def test_answer_span_must_be_a_suffix():
    with pytest.raises(ValueError):
        answer_span([-100, 5, -100, 6])


def test_lr_schedule_warms_up_then_decays_to_zero():
    assert lr_at(0, 100, 10, 1.0) == pytest.approx(0.1)
    assert lr_at(9, 100, 10, 1.0) == pytest.approx(1.0)
    assert lr_at(10, 100, 10, 1.0) == pytest.approx(1.0)
    assert lr_at(55, 100, 10, 1.0) == pytest.approx(0.5 * (1 + math.cos(math.pi * 0.5)))
    assert lr_at(100, 100, 10, 1.0) == pytest.approx(0.0)
