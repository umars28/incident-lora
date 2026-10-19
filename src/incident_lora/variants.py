from __future__ import annotations

import copy
import hashlib
import random
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

INFRA_MARKERS = (
    "postgres",
    "redis",
    "kafka",
    "elasticsearch",
    "coredns",
    "smtp",
    "mysql",
    "etcd",
    "prometheus",
    "kube-",
)

NAME_PREFIXES = (
    "basket", "ledger", "quote", "route", "wallet", "promo", "atlas", "fleet",
    "invoice", "rating", "signup", "session", "tracking", "refund", "stock",
    "booking", "tariff", "voucher", "partner", "merchant", "dispatch", "courier",
    "review", "listing", "returns", "consent", "identity", "feeds", "insight",
    "ticket", "rewards", "settle", "kyc", "payout", "storage", "beacon", "orbit",
    "harbor", "summit", "cedar", "maple", "falcon", "lumen", "nimbus", "quartz",
)

NOISE_LINES = (
    ("INFO", "GET /healthz 200 (elapsed=2ms)"),
    ("INFO", "GET /readyz 200 (elapsed=1ms)"),
    ("DEBUG", "metrics scrape served in 3ms"),
    ("INFO", "config watcher: no changes detected"),
    ("DEBUG", "tls session cache hit ratio 0.97"),
    ("INFO", "background compaction finished in 118ms"),
    ("DEBUG", "feature flags refreshed (0 changed)"),
    ("INFO", "leader lease renewed"),
    ("DEBUG", "dns cache refreshed (42 entries)"),
    ("INFO", "otel exporter flushed 512 spans"),
)

BENIGN_DEPLOY_SUMMARIES = (
    "Docs only: update README and runbook links",
    "Bump base image patch version, no code change",
    "Add unit tests for date parsing",
    "Rename internal metric labels in dashboards repo, no runtime change",
    "Update CODEOWNERS",
)

AUTHORS = ("ari.k", "budi.s", "chen.l", "dina.r", "eko.p", "fatma.y", "gita.n", "hana.m")

VERSION_RE = re.compile(r"\bv(\d+)\.(\d+)\.(\d+)\b")
NODE_RE = re.compile(r"\bip-(\d+)-(\d+)-(\d+)-(\d+)\b")
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\d])")
PROTECTED_SERIES = re.compile(r"size|max|limit|replica|capacity|quota|threshold|budget")


def variant_seed(base_seed: int, scenario_id: str, split: str, index: int) -> int:
    key = f"{base_seed}:{scenario_id}:{split}:{index}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def _walk(value: Any, fn: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, list):
        return [_walk(v, fn) for v in value]
    if isinstance(value, dict):
        return {(fn(k) if isinstance(k, str) else k): _walk(v, fn) for k, v in value.items()}
    return value


def _all_text(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, list):
        for v in value:
            found.extend(_all_text(v))
    elif isinstance(value, dict):
        for k, v in value.items():
            if isinstance(k, str):
                found.append(k)
            found.extend(_all_text(v))
    return found


def service_names(raw: dict[str, Any]) -> list[str]:
    names = {s["name"] for s in raw["world"].get("services", [])}
    names.add(raw["alert"]["service"])
    return sorted(names)


def renameable(name: str) -> bool:
    lowered = name.lower()
    return "-" in name and not any(m in lowered for m in INFRA_MARKERS)


def _rename_alert(raw: dict[str, Any], mapping: dict[str, str]) -> None:
    name = str(raw["alert"]["name"])
    for old, new in mapping.items():
        old_head, new_head = old.split("-")[0], new.split("-")[0]
        name = name.replace(old_head.capitalize(), new_head.capitalize())
    raw["alert"]["name"] = name


def _rename_map(raw: dict[str, Any], rng: random.Random) -> dict[str, str]:
    text = " ".join(_all_text(raw)).lower()
    pool = [p for p in NAME_PREFIXES if p not in text]
    rng.shuffle(pool)
    mapping: dict[str, str] = {}
    for name in service_names(raw):
        if not renameable(name) or not pool:
            continue
        head, _, tail = name.partition("-")
        mapping[name] = f"{pool.pop()}-{tail}"
    return mapping


def _replacer(mapping: dict[str, str]) -> Callable[[str], str]:
    if not mapping:
        return lambda s: s
    keys = sorted(mapping, key=len, reverse=True)
    pattern = re.compile(
        r"(?<![\w-])(" + "|".join(re.escape(k) for k in keys) + r")(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )
    lowered = {k.lower(): v for k, v in mapping.items()}
    return lambda s: pattern.sub(lambda m: lowered[m.group(1).lower()], s)


def _version_shift(rng: random.Random) -> Callable[[str], str]:
    delta = rng.randint(1, 6)
    return lambda s: VERSION_RE.sub(
        lambda m: f"v{int(m.group(1)) + delta}.{m.group(2)}.{m.group(3)}", s
    )


def _node_shift(raw: dict[str, Any], rng: random.Random) -> Callable[[str], str]:
    seen = sorted({m.group(0) for t in _all_text(raw) for m in NODE_RE.finditer(t)})
    mapping: dict[str, str] = {}
    used: set[str] = set(seen)
    for node in seen:
        while True:
            candidate = f"ip-10-{rng.randint(1, 30)}-{rng.randint(0, 31)}-{rng.randint(2, 250)}"
            if candidate not in used:
                used.add(candidate)
                mapping[node] = candidate
                break
    if not mapping:
        return lambda s: s
    return lambda s: NODE_RE.sub(lambda m: mapping.get(m.group(0), m.group(0)), s)


def _commit_shift(raw: dict[str, Any], rng: random.Random) -> Callable[[str], str]:
    mapping = {
        str(d["commit"]): "".join(rng.choice("0123456789abcdef") for _ in range(7))
        for d in raw["world"].get("deploys", [])
        if d.get("commit")
    }
    if not mapping:
        return lambda s: s
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in mapping) + r")\b")
    return lambda s: pattern.sub(lambda m: mapping[m.group(1)], s)


def _protected_numbers(raw: dict[str, Any]) -> set[str]:
    world = raw["world"]
    sources = [raw["alert"], raw["ground_truth"], world.get("logs", {}),
               world.get("k8s", []), world.get("runbooks", []), world.get("deploys", [])]
    found: set[str] = set()
    for src in sources:
        for text in _all_text(src):
            found.update(NUMBER_RE.findall(text))
    return found


def _jitter_metrics(raw: dict[str, Any], rng: random.Random) -> None:
    protected = _protected_numbers(raw)
    for series in raw["world"].get("metrics", {}).values():
        for name, body in series.items():
            points = body.get("points", [])
            values = [p["v"] for p in points]
            if len(set(values)) < 2 or PROTECTED_SERIES.search(name):
                continue
            if any(_num_str(v) in protected or _short(v) in protected for v in values):
                continue
            factor = rng.uniform(0.95, 1.05)
            for p in points:
                p["v"] = _scale(p["v"], factor)


def _num_str(v: Any) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _short(v: Any) -> str:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or abs(v) < 1000:
        return ""
    return str(round(v / 1000, 1))


def _scale(v: Any, factor: float) -> Any:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return v
    if isinstance(v, int):
        return int(round(v * factor))
    decimals = len(repr(v).split(".")[1]) if "." in repr(v) else 0
    return round(v * factor, max(decimals, 1))


def _add_log_noise(raw: dict[str, Any], rng: random.Random) -> None:
    for lines in raw["world"].get("logs", {}).values():
        for _ in range(rng.randint(0, 3)):
            level, msg = rng.choice(NOISE_LINES)
            entry: dict[str, Any] = {"t": -rng.randint(1, 90), "level": level, "msg": msg}
            if rng.random() < 0.5:
                entry["count"] = rng.randint(2, 4000)
            lines.append(entry)


def _add_distractor_deploy(raw: dict[str, Any], rng: random.Random) -> None:
    world = raw["world"]
    alerting = raw["alert"]["service"]
    candidates = [s["name"] for s in world.get("services", []) if s["name"] != alerting]
    if not candidates or rng.random() < 0.5:
        return
    world.setdefault("deploys", []).append(
        {
            "t": -rng.randint(120, 400),
            "service": rng.choice(candidates),
            "version": f"v{rng.randint(1, 9)}.{rng.randint(0, 40)}.{rng.randint(0, 9)}",
            "commit": "".join(rng.choice("0123456789abcdef") for _ in range(7)),
            "author": rng.choice(AUTHORS),
            "summary": rng.choice(BENIGN_DEPLOY_SUMMARIES),
        }
    )


def _shuffle_dict(d: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    keys = list(d)
    rng.shuffle(keys)
    return {k: d[k] for k in keys}


def _reorder(raw: dict[str, Any], rng: random.Random) -> None:
    world = raw["world"]
    for key in ("services", "deploys", "k8s", "runbooks"):
        rng.shuffle(world.get(key, []))
    world["logs"] = _shuffle_dict(world.get("logs", {}), rng)
    world["metrics"] = {
        svc: _shuffle_dict(series, rng)
        for svc, series in _shuffle_dict(world.get("metrics", {}), rng).items()
    }


def _shift_clock(raw: dict[str, Any], rng: random.Random) -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    moment = start + timedelta(minutes=rng.randint(0, 364 * 24 * 60))
    raw["alert"]["fired_at"] = moment.strftime("%Y-%m-%dT%H:%M:00Z")


def make_variant(raw: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    out = copy.deepcopy(raw)
    names = _rename_map(out, rng)
    _rename_alert(out, names)
    rename = _replacer(names)
    version = _version_shift(rng)
    node = _node_shift(out, rng)
    commit = _commit_shift(out, rng)
    keep = {k: out[k] for k in ("id", "category")}
    out = _walk(out, lambda s: version(node(commit(rename(s)))))
    out.update(keep)
    _jitter_metrics(out, rng)
    _add_log_noise(out, rng)
    _add_distractor_deploy(out, rng)
    _reorder(out, rng)
    _shift_clock(out, rng)
    return out
