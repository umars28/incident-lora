import json

from incident_lora.render import render_incident, target_answer
from incident_lora.variants import make_variant, variant_seed


def _by_id(scenarios, sid):
    return next(s for s in scenarios if s.id == sid)


def test_variant_is_deterministic(scenarios):
    raw = _by_id(scenarios, "SC-001").raw
    assert make_variant(raw, 7) == make_variant(raw, 7)
    assert make_variant(raw, 7) != make_variant(raw, 8)


def test_variant_does_not_mutate_source(scenarios):
    raw = _by_id(scenarios, "SC-001").raw
    before = json.dumps(raw, sort_keys=True, default=str)
    make_variant(raw, 3)
    assert json.dumps(raw, sort_keys=True, default=str) == before


def test_rename_is_consistent_across_prompt_and_labels(scenarios):
    raw = _by_id(scenarios, "SC-001").raw
    v = make_variant(raw, 11)
    new_name = v["alert"]["service"]
    assert new_name != "checkout-api"
    assert new_name.endswith("-api")
    text = render_incident(v)
    assert "checkout-api" not in text
    assert new_name in text
    assert new_name in v["ground_truth"]["root_cause"]
    assert "Checkout" not in v["alert"]["name"]


def test_infrastructure_names_are_kept(scenarios):
    v = make_variant(_by_id(scenarios, "SC-001").raw, 5)
    assert "postgres-primary" in render_incident(v)


def test_versions_shift_consistently(scenarios):
    v = make_variant(_by_id(scenarios, "SC-001").raw, 13)
    shifted = [d["version"] for d in v["world"]["deploys"] if d["service"] == v["alert"]["service"]]
    assert len(shifted) == 1 and shifted[0] != "v2.31.0"
    groups = [alt for g in v["ground_truth"]["must_include"] for alt in g]
    assert shifted[0] in groups
    assert shifted[0] in v["ground_truth"]["root_cause"]
    assert shifted[0] in render_incident(v)


def test_node_names_shift_consistently(scenarios):
    v = make_variant(_by_id(scenarios, "SC-003").raw, 21)
    groups = [alt for g in v["ground_truth"]["must_include"] for alt in g]
    nodes = [g for g in groups if g.startswith("ip-")]
    assert nodes and nodes[0] != "ip-10-4-9-9"
    assert nodes[0] in render_incident(v)


def test_numbers_cited_elsewhere_are_not_jittered(scenarios):
    raw = _by_id(scenarios, "SC-001").raw
    for seed in range(20):
        v = make_variant(raw, seed)
        svc = v["alert"]["service"]
        p99 = [p["v"] for p in v["world"]["metrics"][svc]["http_p99_latency_ms"]["points"]]
        pool = [p["v"] for p in v["world"]["metrics"][svc]["db_pool_in_use"]["points"]]
        assert 4210 in p99
        assert pool == [6, 20]


def test_uncited_series_are_jittered_within_five_percent(scenarios):
    raw = _by_id(scenarios, "SC-001").raw
    changed = False
    for seed in range(20):
        v = make_variant(raw, seed)
        svc = v["alert"]["service"]
        values = [p["v"] for p in v["world"]["metrics"][svc]["queue_depth"]["points"]]
        assert 1111 * 0.95 - 1 <= values[0] <= 1111 * 1.05 + 1
        changed = changed or values != [1111, 7777]
    assert changed


def test_label_survives_variation(scenarios):
    raw = _by_id(scenarios, "SC-001").raw
    v = make_variant(raw, 4)
    assert target_answer(v)["category"] == "resource-exhaustion"
    assert target_answer(v)["action_id"] == "rollback_deploy"


def test_variant_seed_depends_on_split():
    assert variant_seed(1, "SC-001", "train", 1) != variant_seed(1, "SC-001", "test_heldout", 1)
    assert variant_seed(1, "SC-001", "train", 1) == variant_seed(1, "SC-001", "train", 1)
