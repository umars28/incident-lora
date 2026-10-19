from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from .scoring import _percentile


async def _one(client: httpx.AsyncClient, url: str, body: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        resp = await client.post(url, json=body)
        elapsed = time.perf_counter() - started
        if resp.status_code != 200:
            return {"ok": False, "latency_s": elapsed, "status": resp.status_code, "tokens": 0}
        usage = resp.json().get("usage") or {}
        return {"ok": True, "latency_s": elapsed, "status": 200,
                "tokens": int(usage.get("completion_tokens", 0) or 0),
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0)}
    except httpx.HTTPError as exc:
        return {"ok": False, "latency_s": time.perf_counter() - started,
                "status": type(exc).__name__, "tokens": 0}


async def run_load(base_url: str, model: str, prompts: list[list[dict[str, str]]],
                   concurrency: int, total: int, max_tokens: int = 384,
                   timeout: float = 900.0, transport: httpx.AsyncBaseTransport | None = None
                   ) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(total):
        queue.put_nowait(i)
    results: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
        async def worker() -> None:
            while True:
                try:
                    i = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                body = {"model": model, "messages": prompts[i % len(prompts)],
                        "temperature": 0, "max_tokens": max_tokens}
                results.append(await _one(client, url, body))

        started = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - started

    ok = [r for r in results if r["ok"]]
    lat = [r["latency_s"] for r in ok]
    tokens = sum(r["tokens"] for r in ok)
    return {
        "model": model,
        "concurrency": concurrency,
        "requests": total,
        "succeeded": len(ok),
        "failed": total - len(ok),
        "wall_s": round(wall, 3),
        "requests_per_s": round(len(ok) / wall, 4) if wall else 0.0,
        "output_tokens": tokens,
        "output_tokens_per_s": round(tokens / wall, 2) if wall else 0.0,
        "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in ok),
        "latency_p50_s": round(_percentile(lat, 0.5), 3),
        "latency_p95_s": round(_percentile(lat, 0.95), 3),
        "latency_max_s": round(max(lat), 3) if lat else 0.0,
        "failures": sorted({str(r["status"]) for r in results if not r["ok"]}),
    }
