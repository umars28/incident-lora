import json

from incident_lora.dataset import build_dataset, read_jsonl
from incident_lora.split import holdout_split


def test_holdout_rule_picks_highest_id_per_multi_member_category(scenarios):
    split = holdout_split(scenarios)
    assert split.heldout == ("SC-002", "SC-004")
    assert "SC-005" in split.train
    assert set(split.train).isdisjoint(split.heldout)


def test_build_writes_disjoint_sets(tmp_path, scenarios, actions):
    manifest = build_dataset(scenarios, actions, tmp_path, train_variants=4,
                             test_variants=3, seed=1)
    train = read_jsonl(tmp_path / "train.jsonl")
    heldout = read_jsonl(tmp_path / "test_heldout.jsonl")
    assert manifest["counts"] == {"train": 12, "val": 3, "test_seen": 3, "test_heldout": 6}
    assert {r["scenario_id"] for r in train} == {"SC-001", "SC-003", "SC-005"}
    assert {r["scenario_id"] for r in heldout} == {"SC-002", "SC-004"}
    checks = manifest["leak_checks"]
    assert checks["scenario_overlap"] == []
    assert checks["heldout_prompts_seen_in_train"] == 0
    assert checks["seen_prompts_seen_in_train"] == 0
    assert checks["val_prompts_seen_in_train"] == 0


def test_original_scenario_is_included_once(tmp_path, scenarios, actions):
    build_dataset(scenarios, actions, tmp_path, train_variants=3, test_variants=2, seed=1)
    heldout = read_jsonl(tmp_path / "test_heldout.jsonl")
    originals = [r for r in heldout if r["id"].endswith("-orig")]
    assert len(originals) == 2
    assert any("orders-api" in r["messages"][1]["content"] for r in originals)


def test_build_is_reproducible(tmp_path, scenarios, actions):
    a, b = tmp_path / "a", tmp_path / "b"
    build_dataset(scenarios, actions, a, 3, 2, seed=9)
    build_dataset(scenarios, actions, b, 3, 2, seed=9)
    for name in ("train", "val", "test_seen", "test_heldout"):
        assert (a / f"{name}.jsonl").read_text() == (b / f"{name}.jsonl").read_text()


def test_target_is_valid_json_with_the_label(tmp_path, scenarios, actions):
    build_dataset(scenarios, actions, tmp_path, 2, 2, seed=1)
    for r in read_jsonl(tmp_path / "train.jsonl"):
        target = json.loads(r["target"])
        assert target["category"] == r["category"]
        assert target["action_id"] in r["expected_actions"]
        assert set(target) == {"root_cause", "category", "action_id"}
