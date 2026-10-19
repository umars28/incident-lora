from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .scenarios import CATEGORIES

REQUIRED_KEYS = ("root_cause", "category", "action_id")
FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def strict_parse(output: str) -> dict[str, Any] | None:
    text = FENCE_RE.sub("", THINK_RE.sub("", output).strip()).strip()
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def lenient_parse(output: str) -> dict[str, Any] | None:
    strict = strict_parse(output)
    if strict is not None:
        return strict
    text = THINK_RE.sub("", output)
    decoder = json.JSONDecoder()
    for start in (m.start() for m in re.finditer(r"\{", text)):
        try:
            value, _ = decoder.raw_decode(text[start:])
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return None


def schema_valid(obj: dict[str, Any] | None, action_ids: set[str]) -> bool:
    if obj is None or set(obj) != set(REQUIRED_KEYS):
        return False
    if not all(isinstance(obj[k], str) and obj[k].strip() for k in REQUIRED_KEYS):
        return False
    return obj["category"] in CATEGORIES and obj["action_id"] in action_ids


def _group_hit(haystack: str, group: list[str] | str) -> str | None:
    options = group if isinstance(group, list) else [group]
    for alternative in options:
        if alternative.lower() in haystack:
            return alternative
    return None


@dataclass
class ItemScore:
    id: str
    scenario_id: str
    json_valid: bool = False
    schema_valid: bool = False
    category_correct: bool = False
    root_cause_hit: bool = False
    matched_groups: int = 0
    total_groups: int = 0
    clean: bool = True
    misleading_claims: list[str] = field(default_factory=list)
    action_match: bool = False
    predicted_category: str | None = None
    predicted_action: str | None = None
    latency_s: float = 0.0
    output_tokens: int = 0
    error: str | None = None

    @property
    def strict_correct(self) -> bool:
        return self.root_cause_hit and self.action_match and self.clean

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["strict_correct"] = self.strict_correct
        return d


def score_output(record: dict[str, Any], output: str, action_ids: set[str],
                 latency_s: float = 0.0, output_tokens: int = 0,
                 error: str | None = None) -> ItemScore:
    must = record.get("must_include", [])
    score = ItemScore(
        id=record["id"],
        scenario_id=record["scenario_id"],
        total_groups=len(must),
        latency_s=latency_s,
        output_tokens=output_tokens,
        error=error,
    )
    strict = strict_parse(output)
    score.json_valid = strict is not None
    score.schema_valid = schema_valid(strict, action_ids)
    obj = lenient_parse(output)
    if obj is None:
        return score

    category = obj.get("category")
    action = obj.get("action_id")
    score.predicted_category = category if isinstance(category, str) else None
    score.predicted_action = action if isinstance(action, str) else None
    score.category_correct = score.predicted_category == record["category"]
    score.action_match = score.predicted_action in record.get("expected_actions", [])

    haystack = _normalise(str(obj.get("root_cause", "")))
    if not haystack.strip():
        return score
    score.matched_groups = sum(1 for g in must if _group_hit(haystack, g))
    score.root_cause_hit = bool(must) and score.matched_groups == len(must)
    for group in record.get("must_not_include", []):
        hit = _group_hit(haystack, group)
        if hit:
            score.misleading_claims.append(hit)
    score.clean = not score.misleading_claims
    return score


def _pct(n: int, d: int) -> float:
    return round(100 * n / d, 1) if d else 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarise(scores: list[ItemScore]) -> dict[str, Any]:
    n = len(scores)
    latencies = [s.latency_s for s in scores if s.error is None]
    tokens = sum(s.output_tokens for s in scores if s.error is None)
    by_scenario: dict[str, dict[str, int]] = {}
    for s in scores:
        b = by_scenario.setdefault(s.scenario_id, {"n": 0, "root_cause_hit": 0,
                                                   "category_correct": 0, "action_match": 0})
        b["n"] += 1
        b["root_cause_hit"] += int(s.root_cause_hit)
        b["category_correct"] += int(s.category_correct)
        b["action_match"] += int(s.action_match)

    def count(attr: str) -> int:
        return sum(1 for s in scores if getattr(s, attr))

    return {
        "n": n,
        "errors": sum(1 for s in scores if s.error),
        "json_valid": count("json_valid"),
        "json_valid_pct": _pct(count("json_valid"), n),
        "schema_valid": count("schema_valid"),
        "schema_valid_pct": _pct(count("schema_valid"), n),
        "root_cause_hit": count("root_cause_hit"),
        "root_cause_hit_pct": _pct(count("root_cause_hit"), n),
        "mean_groups_matched_pct": _pct(sum(s.matched_groups for s in scores),
                                        sum(s.total_groups for s in scores)),
        "category_correct": count("category_correct"),
        "category_correct_pct": _pct(count("category_correct"), n),
        "action_match": count("action_match"),
        "action_match_pct": _pct(count("action_match"), n),
        "clean": count("clean"),
        "strict_correct": count("strict_correct"),
        "strict_correct_pct": _pct(count("strict_correct"), n),
        "latency_p50_s": round(_percentile(latencies, 0.5), 3),
        "latency_p95_s": round(_percentile(latencies, 0.95), 3),
        "latency_mean_s": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "output_tokens_total": tokens,
        "output_tokens_per_s": round(tokens / sum(latencies), 2) if latencies and sum(latencies) else 0.0,
        "by_scenario": dict(sorted(by_scenario.items())),
    }
