import asyncio
import json
import threading
from http.server import ThreadingHTTPServer

import httpx

from incident_lora.loadtest import run_load
from incident_lora.messages_shim import chat_to_messages, make_handler, messages_to_chat


def test_messages_to_chat_maps_system_and_blocks():
    chat = messages_to_chat({
        "model": "incident-lora",
        "system": [{"type": "text", "text": "be terse"}],
        "max_tokens": 200,
        "temperature": 0,
        "stop_sequences": ["\n\n"],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "alert"}]}],
    })
    assert chat == {
        "model": "incident-lora",
        "messages": [{"role": "system", "content": "be terse"},
                     {"role": "user", "content": "alert"}],
        "max_tokens": 200,
        "temperature": 0,
        "stop": ["\n\n"],
    }


def test_chat_to_messages_carries_usage_for_cost_attribution():
    out = chat_to_messages({
        "id": "cmpl-1",
        "choices": [{"message": {"content": "{\"a\":1}"}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 1500, "completion_tokens": 90},
    }, "incident-lora")
    assert out["content"] == [{"type": "text", "text": "{\"a\":1}"}]
    assert out["stop_reason"] == "max_tokens"
    assert out["usage"] == {"input_tokens": 1500, "output_tokens": 90}


def _fake_vllm(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    return httpx.Response(200, json={
        "id": "x",
        "choices": [{"message": {"content": body["model"]}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    })


def test_shim_end_to_end_over_http():
    upstream = httpx.Client(transport=httpx.MockTransport(_fake_vllm))
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler("http://vllm:8000", upstream))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        resp = httpx.post(f"http://127.0.0.1:{port}/v1/messages", json={
            "model": "incident-lora", "max_tokens": 50,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 200
        assert resp.json()["content"][0]["text"] == "incident-lora"
        assert httpx.post(f"http://127.0.0.1:{port}/v1/other", json={}).status_code == 404
        assert httpx.post(f"http://127.0.0.1:{port}/v1/messages", content=b"nope").status_code == 400
    finally:
        server.shutdown()


def test_loadtest_reports_throughput_and_percentiles():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 3:
            return httpx.Response(503, json={})
        return httpx.Response(200, json={"usage": {"completion_tokens": 5, "prompt_tokens": 100}})

    result = asyncio.run(run_load("http://vllm:8000", "incident-lora",
                                  [[{"role": "user", "content": "x"}]], concurrency=2,
                                  total=6, transport=httpx.MockTransport(handler)))
    assert result["succeeded"] == 5 and result["failed"] == 1
    assert result["output_tokens"] == 25
    assert result["failures"] == ["503"]
    assert result["latency_p95_s"] >= result["latency_p50_s"]
