from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .backends import Backend
from .scoring import ItemScore, score_output, summarise


def run_eval(backend: Backend, records: list[dict[str, Any]], action_ids: set[str],
             progress: Callable[[int, int, ItemScore], None] | None = None,
             keep_outputs: bool = True) -> dict[str, Any]:
    scores: list[ItemScore] = []
    outputs: list[dict[str, Any]] = []
    for i, record in enumerate(records):
        try:
            gen = backend.generate(record["messages"])
            score = score_output(record, gen.text, action_ids, gen.latency_s, gen.output_tokens)
            text = gen.text
        except Exception as exc:
            score = score_output(record, "", action_ids, error=f"{type(exc).__name__}: {exc}")
            text = ""
        scores.append(score)
        if keep_outputs:
            outputs.append({**score.to_dict(), "output": text})
        if progress:
            progress(i + 1, len(records), score)
    return {
        "backend": backend.name,
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "python": sys.version.split()[0]},
        "summary": summarise(scores),
        "items": outputs,
    }


def write_result(result: dict[str, Any], path: Path, extra: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {**(extra or {}), **result}
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n")


COLUMNS = (
    ("root cause hit", "root_cause_hit"),
    ("category", "category_correct"),
    ("action", "action_match"),
    ("strict correct", "strict_correct"),
    ("valid JSON", "json_valid"),
    ("valid schema", "schema_valid"),
)


def markdown_table(results: list[dict[str, Any]]) -> str:
    head = ["model", "test set", "n"] + [c[0] for c in COLUMNS] + ["p50 s", "p95 s", "tok/s"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in results:
        s = r["summary"]
        n = s["n"]
        cells = [r.get("label", r["backend"]), r.get("test_set", ""), str(n)]
        cells += [f"{s[key]}/{n} ({s[key + '_pct']}%)" for _, key in COLUMNS]
        cells += [str(s["latency_p50_s"]), str(s["latency_p95_s"]), str(s["output_tokens_per_s"])]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def per_scenario_table(results: list[dict[str, Any]]) -> str:
    labels = [r.get("label", r["backend"]) for r in results]
    scenarios = sorted({sid for r in results for sid in r["summary"]["by_scenario"]})
    head = ["scenario"] + labels
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for sid in scenarios:
        row = [sid]
        for r in results:
            b = r["summary"]["by_scenario"].get(sid)
            row.append(
                f"{b['root_cause_hit']}/{b['n']} rc, {b['category_correct']}/{b['n']} cat, "
                f"{b['action_match']}/{b['n']} act" if b else "-"
            )
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
