from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIOS_DIR = REPO_ROOT.parent / "llm-platform" / "benchmarks" / "incidents"

CATEGORIES = (
    "resource-exhaustion",
    "dependency-failure",
    "capacity",
    "config-error",
    "data-layer",
    "network",
    "cascading-failure",
    "bad-release",
    "false-alarm",
    "misleading-signal",
)


def scenarios_dir(override: str | Path | None = None) -> Path:
    if override:
        return Path(override)
    env = os.environ.get("INCIDENT_SCENARIOS_DIR")
    return Path(env) if env else DEFAULT_SCENARIOS_DIR


@dataclass(frozen=True)
class Scenario:
    id: str
    category: str
    title: str
    raw: dict[str, Any]
    sha256: str

    @property
    def ground_truth(self) -> dict[str, Any]:
        return self.raw["ground_truth"]


def load_scenario(path: Path) -> Scenario:
    data = path.read_bytes()
    raw = yaml.safe_load(data)
    return Scenario(
        id=raw["id"],
        category=raw["category"],
        title=raw["title"],
        raw=raw,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def load_scenarios(directory: str | Path | None = None) -> list[Scenario]:
    root = scenarios_dir(directory)
    paths = sorted(root.glob("SC-*.yaml"))
    if not paths:
        raise FileNotFoundError(
            f"no SC-*.yaml scenarios under {root}; set INCIDENT_SCENARIOS_DIR"
        )
    return [load_scenario(p) for p in paths]


def load_actions(directory: str | Path | None = None) -> list[dict[str, Any]]:
    common = scenarios_dir(directory) / "_common.yaml"
    raw = yaml.safe_load(common.read_text())
    return list(raw["actions"])
