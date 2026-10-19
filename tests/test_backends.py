import json

import httpx

from incident_lora.backends import FakeBackend, OpenAIBackend
from incident_lora.dataset import build_dataset, read_jsonl
from incident_lora.evaluate import markdown_table, run_eval


def test_openai_backend_speaks_chat_completions():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "{}"}}],
            "usage": {"completion_tokens": 7},
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = OpenAIBackend("http://vllm:8000/", "incident-lora", client=client)
    gen = backend.generate([{"role": "user", "content": "hi"}])
    assert seen["url"] == "http://vllm:8000/v1/chat/completions"
    assert seen["body"]["model"] == "incident-lora"
    assert seen["body"]["temperature"] == 0
    assert gen.output_tokens == 7 and gen.text == "{}"


def test_eval_with_oracle_and_broken_backends(tmp_path, scenarios, actions):
    build_dataset(scenarios, actions, tmp_path, 2, 2, seed=1)
    records = read_jsonl(tmp_path / "test_heldout.jsonl")
    by_prompt = {r["messages"][1]["content"]: r["target"] for r in records}
    ids = {a["id"] for a in actions}

    oracle = FakeBackend(lambda m: by_prompt[m[1]["content"]], name="oracle")
    result = run_eval(oracle, records, ids)
    assert result["summary"]["root_cause_hit"] == len(records)
    assert result["summary"]["schema_valid"] == len(records)

    def boom(_):
        raise RuntimeError("down")

    broken = run_eval(FakeBackend(boom, name="broken"), records, ids)
    assert broken["summary"]["errors"] == len(records)
    assert broken["summary"]["root_cause_hit"] == 0

    table = markdown_table([{**result, "label": "oracle", "test_set": "test_heldout"}])
    assert "| oracle | test_heldout | 4 |" in table
