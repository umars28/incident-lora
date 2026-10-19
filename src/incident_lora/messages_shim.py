from __future__ import annotations

import json
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx

STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def messages_to_chat(body: dict[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    system = _text(body.get("system")) if body.get("system") else ""
    if system:
        messages.append({"role": "system", "content": system})
    for m in body.get("messages", []):
        messages.append({"role": m["role"], "content": _text(m.get("content"))})
    chat: dict[str, Any] = {
        "model": body["model"],
        "messages": messages,
        "max_tokens": int(body.get("max_tokens", 1024)),
    }
    for key in ("temperature", "top_p"):
        if key in body:
            chat[key] = body[key]
    if body.get("stop_sequences"):
        chat["stop"] = list(body["stop_sequences"])
    return chat


def chat_to_messages(resp: dict[str, Any], model: str) -> dict[str, Any]:
    choice = (resp.get("choices") or [{}])[0]
    usage = resp.get("usage") or {}
    return {
        "id": "msg_" + str(resp.get("id") or uuid.uuid4().hex),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": (choice.get("message") or {}).get("content") or ""}],
        "stop_reason": STOP_REASONS.get(choice.get("finish_reason") or "stop", "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "output_tokens": int(usage.get("completion_tokens", 0) or 0),
        },
    }


def make_handler(upstream: str, client: httpx.Client) -> type[BaseHTTPRequestHandler]:
    target = upstream.rstrip("/") + "/v1/chat/completions"

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path in ("/healthz", "/readyz"):
                self._send(200, {"ok": True, "upstream": upstream})
            else:
                self._send(404, {"type": "error", "error": {"type": "not_found_error",
                                                            "message": self.path}})

        def do_POST(self) -> None:
            if self.path != "/v1/messages":
                self._send(404, {"type": "error", "error": {"type": "not_found_error",
                                                            "message": self.path}})
                return
            length = int(self.headers.get("content-length", 0))
            try:
                body = json.loads(self.rfile.read(length))
                chat = messages_to_chat(body)
            except (ValueError, KeyError) as exc:
                self._send(400, {"type": "error", "error": {"type": "invalid_request_error",
                                                            "message": str(exc)}})
                return
            try:
                resp = client.post(target, json=chat)
            except httpx.HTTPError as exc:
                self._send(502, {"type": "error", "error": {"type": "api_error",
                                                            "message": type(exc).__name__}})
                return
            if resp.status_code != 200:
                self._send(resp.status_code, {"type": "error", "error": {
                    "type": "api_error", "message": resp.text[:500]}})
                return
            self._send(200, chat_to_messages(resp.json(), body["model"]))

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return Handler


def serve(upstream: str, host: str = "0.0.0.0", port: int = 8090) -> None:
    client = httpx.Client(timeout=900.0)
    server = ThreadingHTTPServer((host, port), make_handler(upstream, client))
    print(f"messages shim on {host}:{port} -> {upstream}/v1/chat/completions", flush=True)
    server.serve_forever()
