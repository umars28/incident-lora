from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import httpx

MAX_NEW_TOKENS = 384


@dataclass
class Generation:
    text: str
    latency_s: float
    output_tokens: int


class Backend(Protocol):
    name: str

    def generate(self, messages: list[dict[str, str]]) -> Generation: ...


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sync(device: str) -> None:
    import torch

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


class HFBackend:
    def __init__(self, model_id: str, adapter: str | None = None, device: str | None = None,
                 max_new_tokens: int = MAX_NEW_TOKENS, name: str | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or pick_device()
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        self.model = model.to(self.device).eval()
        self.name = name or (f"{model_id}+lora" if adapter else model_id)

    def generate(self, messages: list[dict[str, str]]) -> Generation:
        import torch

        enc = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        input_ids = enc["input_ids"].to(self.device)
        attention = enc["attention_mask"].to(self.device)
        _sync(self.device)
        started = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        _sync(self.device)
        elapsed = time.perf_counter() - started
        new = out[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        return Generation(text=text, latency_s=elapsed, output_tokens=int(new.shape[0]))


class OllamaBackend:
    def __init__(self, model: str, host: str | None = None, timeout: float = 600.0,
                 max_new_tokens: int = MAX_NEW_TOKENS) -> None:
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.client = httpx.Client(timeout=timeout)
        self.max_new_tokens = max_new_tokens
        self.name = f"ollama:{model}"

    def generate(self, messages: list[dict[str, str]]) -> Generation:
        started = time.perf_counter()
        resp = self.client.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": self.max_new_tokens, "num_ctx": 8192},
            },
        )
        resp.raise_for_status()
        elapsed = time.perf_counter() - started
        body = resp.json()
        return Generation(
            text=body.get("message", {}).get("content", ""),
            latency_s=elapsed,
            output_tokens=int(body.get("eval_count", 0) or 0),
        )


class OpenAIBackend:
    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 timeout: float = 600.0, max_new_tokens: int = MAX_NEW_TOKENS,
                 client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_new_tokens = max_new_tokens
        headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.Client(timeout=timeout, headers=headers)
        self.name = f"openai:{model}"

    def request_body(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_new_tokens,
        }

    def generate(self, messages: list[dict[str, str]]) -> Generation:
        started = time.perf_counter()
        resp = self.client.post(f"{self.base_url}/v1/chat/completions",
                                json=self.request_body(messages))
        resp.raise_for_status()
        elapsed = time.perf_counter() - started
        body = resp.json()
        usage = body.get("usage") or {}
        return Generation(
            text=body["choices"][0]["message"]["content"] or "",
            latency_s=elapsed,
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
        )


class FakeBackend:
    def __init__(self, respond: Callable[[list[dict[str, str]]], str], name: str = "fake") -> None:
        self.respond = respond
        self.name = name

    def generate(self, messages: list[dict[str, str]]) -> Generation:
        text = self.respond(messages)
        return Generation(text=text, latency_s=0.0, output_tokens=len(text.split()))
