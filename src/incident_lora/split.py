from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .scenarios import Scenario


@dataclass(frozen=True)
class Split:
    train: tuple[str, ...]
    heldout: tuple[str, ...]
    rule: str


HOLDOUT_RULE = (
    "for every category with at least two scenarios, the highest-numbered scenario "
    "is held out; categories with a single scenario stay in training"
)


def holdout_split(scenarios: list[Scenario]) -> Split:
    by_category: dict[str, list[str]] = defaultdict(list)
    for s in scenarios:
        by_category[s.category].append(s.id)
    heldout = sorted(
        max(ids) for ids in by_category.values() if len(ids) >= 2
    )
    train = sorted(s.id for s in scenarios if s.id not in heldout)
    return Split(train=tuple(train), heldout=tuple(heldout), rule=HOLDOUT_RULE)
