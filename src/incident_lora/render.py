from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from .scenarios import CATEGORIES

SYSTEM_TEMPLATE = """You are an on-call SRE. You receive a frozen snapshot of a production incident: the alert, recent deploys, logs, metrics, Kubernetes state and runbooks. Find the root cause. The alerting service is often a victim, and every snapshot contains misleading signals.

Answer with one JSON object and nothing else, using exactly these keys:
{{"root_cause": "<two to four sentences naming the cause, the mechanism and the evidence>", "category": "<one category>", "action_id": "<one action id>"}}

Categories: {categories}

Actions:
{actions}"""


def system_prompt(actions: list[dict[str, Any]]) -> str:
    lines = [f"- {a['id']}: {a['description'].strip()}" for a in actions]
    return SYSTEM_TEMPLATE.format(
        categories=", ".join(CATEGORIES), actions="\n".join(lines)
    )


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _clock(base: datetime, offset_min: float) -> str:
    return (base + timedelta(minutes=offset_min)).strftime("%H:%M")


def _num(value: Any) -> str:
    if isinstance(value, float) and value.is_integer() and abs(value) >= 1000:
        return str(int(value))
    return str(value)


def _compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=str)


def render_incident(raw: dict[str, Any]) -> str:
    alert = raw["alert"]
    world = raw["world"]
    base = _parse_time(str(alert["fired_at"]))
    out: list[str] = []

    out.append("## Alert")
    out.append(
        f"{alert['name']} severity={alert['severity']} service={alert['service']} "
        f"fired_at={base.strftime('%Y-%m-%d %H:%M')}Z"
    )
    out.append(" ".join(str(alert["description"]).split()))

    out.append("\n## Services")
    for svc in world.get("services", []):
        deps = ", ".join(svc.get("depends_on", [])) or "none"
        out.append(
            f"- {svc['name']} tier={svc.get('tier')} owner={svc.get('owner')} depends_on=[{deps}]"
        )

    out.append("\n## Deploys")
    deploys = world.get("deploys", [])
    if not deploys:
        out.append("none")
    for d in deploys:
        out.append(
            f"- {_clock(base, d['t'])} {d['service']} {d.get('version', '')} "
            f"commit={d.get('commit', '')} by={d.get('author', '')}: {d.get('summary', '')}"
        )

    out.append("\n## Logs")
    for service, lines in world.get("logs", {}).items():
        out.append(f"[{service}]")
        for line in sorted(lines, key=lambda x: x["t"]):
            count = f" (x{line['count']})" if line.get("count") else ""
            out.append(f"{_clock(base, line['t'])} {line['level']} {line['msg']}{count}")

    out.append("\n## Metrics")
    for service, series in world.get("metrics", {}).items():
        for name, body in series.items():
            points = " ".join(
                f"{_clock(base, p['t'])}={_num(p['v'])}" for p in body.get("points", [])
            )
            out.append(f"{service} {name} ({body.get('unit', '')}): {points}")

    out.append("\n## Kubernetes")
    for obj in world.get("k8s", []):
        head = f"{obj.get('kind')} {obj.get('namespace', '-')}/{obj.get('name')}"
        out.append(head)
        if obj.get("spec"):
            out.append(f"  spec {_compact(obj['spec'])}")
        if obj.get("status"):
            out.append(f"  status {_compact(obj['status'])}")
        for ev in obj.get("events", []) or []:
            out.append(
                f"  event {_clock(base, ev['t'])} {ev.get('type')} {ev.get('reason')}: {ev.get('msg')}"
            )

    out.append("\n## Runbooks")
    for rb in world.get("runbooks", []):
        out.append(f"{rb['id']} {rb['title']}: {' '.join(str(rb['body']).split())}")

    return "\n".join(out)


def build_messages(raw: dict[str, Any], actions: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt(actions)},
        {"role": "user", "content": render_incident(raw)},
    ]


def target_answer(raw: dict[str, Any]) -> dict[str, str]:
    truth = raw["ground_truth"]
    return {
        "root_cause": " ".join(str(truth["root_cause"]).split()),
        "category": raw["category"],
        "action_id": truth["expected_actions"][0],
    }


def target_text(raw: dict[str, Any]) -> str:
    return json.dumps(target_answer(raw), ensure_ascii=False)
