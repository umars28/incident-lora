from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .render import build_messages, target_answer, target_text
from .scenarios import Scenario
from .split import Split, holdout_split
from .variants import make_variant, variant_seed


def make_record(raw: dict[str, Any], actions: list[dict[str, Any]], variant_id: str,
                split: str) -> dict[str, Any]:
    truth = raw["ground_truth"]
    return {
        "id": variant_id,
        "scenario_id": raw["id"],
        "split": split,
        "category": raw["category"],
        "messages": build_messages(raw, actions),
        "target": target_text(raw),
        "answer": target_answer(raw),
        "expected_actions": list(truth.get("expected_actions", [])),
        "must_include": truth.get("must_include", []),
        "must_not_include": truth.get("must_not_include", []),
    }


def _variants(scenario: Scenario, split: str, count: int, seed: int,
              include_original: bool) -> Iterable[tuple[str, dict[str, Any]]]:
    for index in range(count):
        if include_original and index == 0:
            yield f"{scenario.id}-{split}-orig", scenario.raw
            continue
        yield (
            f"{scenario.id}-{split}-{index:03d}",
            make_variant(scenario.raw, variant_seed(seed, scenario.id, split, index)),
        )


def _ngrams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = text.lower().split()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def leak_report(sets: dict[str, list[dict[str, Any]]], split: Split,
                scenarios: list[Scenario]) -> dict[str, Any]:
    train = sets["train"]
    train_ids = {r["scenario_id"] for r in train}
    heldout_ids = {r["scenario_id"] for r in sets["test_heldout"]}
    train_prompts = {r["messages"][1]["content"] for r in train}
    train_targets = [r["answer"]["root_cause"] for r in train]
    train_grams = set().union(*(_ngrams(t) for t in train_targets)) if train_targets else set()

    by_id = {s.id: s for s in scenarios}
    train_runbooks = {
        rb["id"] for sid in split.train for rb in by_id[sid].raw["world"].get("runbooks", [])
    }
    shared_runbooks = sorted(
        {
            rb["id"]
            for sid in split.heldout
            for rb in by_id[sid].raw["world"].get("runbooks", [])
            if rb["id"] in train_runbooks
        }
    )

    overlaps = []
    for r in sets["test_heldout"]:
        grams = _ngrams(r["answer"]["root_cause"])
        if grams:
            overlaps.append(len(grams & train_grams) / len(grams))

    return {
        "scenario_overlap": sorted(train_ids & heldout_ids),
        "heldout_prompts_seen_in_train": sum(
            1 for r in sets["test_heldout"] if r["messages"][1]["content"] in train_prompts
        ),
        "seen_prompts_seen_in_train": sum(
            1 for r in sets["test_seen"] if r["messages"][1]["content"] in train_prompts
        ),
        "val_prompts_seen_in_train": sum(
            1 for r in sets["val"] if r["messages"][1]["content"] in train_prompts
        ),
        "heldout_target_5gram_overlap_max": round(max(overlaps), 4) if overlaps else 0.0,
        "heldout_target_5gram_overlap_mean": round(sum(overlaps) / len(overlaps), 4) if overlaps else 0.0,
        "runbook_ids_shared_between_heldout_and_train": shared_runbooks,
    }


def build_dataset(scenarios: list[Scenario], actions: list[dict[str, Any]], out_dir: Path,
                  train_variants: int, test_variants: int, seed: int) -> dict[str, Any]:
    split = holdout_split(scenarios)
    by_id = {s.id: s for s in scenarios}
    sets: dict[str, list[dict[str, Any]]] = {
        "train": [], "val": [], "test_seen": [], "test_heldout": []
    }
    plan = [
        ("train", split.train, train_variants, True),
        ("val", split.train, 1, False),
        ("test_seen", split.train, 1, False),
        ("test_heldout", split.heldout, test_variants, True),
    ]
    for name, ids, count, original in plan:
        for sid in ids:
            for vid, raw in _variants(by_id[sid], name, count, seed, original):
                sets[name].append(make_record(raw, actions, vid, name))

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, records in sets.items():
        with (out_dir / f"{name}.jsonl").open("w") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "seed": seed,
        "train_variants_per_scenario": train_variants,
        "test_variants_per_scenario": test_variants,
        "split_rule": split.rule,
        "train_scenarios": list(split.train),
        "heldout_scenarios": list(split.heldout),
        "heldout_categories": {sid: by_id[sid].category for sid in split.heldout},
        "counts": {k: len(v) for k, v in sets.items()},
        "scenario_sha256": {s.id: s.sha256 for s in scenarios},
        "leak_checks": leak_report(sets, split, scenarios),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]
