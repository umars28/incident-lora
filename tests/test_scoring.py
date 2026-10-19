import json

from incident_lora.scoring import lenient_parse, schema_valid, score_output, strict_parse, summarise

ACTIONS = {"rollback_deploy", "increase_db_pool_size", "drain_node", "no_action_required"}

RECORD = {
    "id": "SC-001-test-000",
    "scenario_id": "SC-001",
    "category": "resource-exhaustion",
    "expected_actions": ["rollback_deploy", "increase_db_pool_size"],
    "must_include": [["connection pool", "pool"], ["saturat", "20/20"], ["v2.31.0", "deploy"]],
    "must_not_include": [["postgres cpu"]],
}


def answer(**kw):
    base = {"root_cause": "The v2.31.0 deploy saturated the connection pool.",
            "category": "resource-exhaustion", "action_id": "rollback_deploy"}
    base.update(kw)
    return json.dumps(base)


def test_perfect_answer():
    s = score_output(RECORD, answer(), ACTIONS)
    assert s.json_valid and s.schema_valid and s.root_cause_hit
    assert s.category_correct and s.action_match and s.clean and s.strict_correct


def test_fenced_json_is_strict_valid():
    assert strict_parse("```json\n" + answer() + "\n```") is not None


def test_prose_around_json_is_not_strict_but_still_scored():
    s = score_output(RECORD, "Here is my answer: " + answer() + " hope it helps", ACTIONS)
    assert not s.json_valid
    assert s.root_cause_hit and s.action_match


def test_think_block_is_ignored():
    assert strict_parse("<think>hmm {not json}</think>" + answer()) is not None


def test_missing_group_fails_root_cause():
    s = score_output(RECORD, answer(root_cause="The pool saturated."), ACTIONS)
    assert s.matched_groups == 2 and not s.root_cause_hit


def test_misleading_claim_breaks_strict():
    s = score_output(RECORD, answer(root_cause="deploy saturated the pool and postgres cpu"), ACTIONS)
    assert s.root_cause_hit and not s.clean and not s.strict_correct


def test_unknown_action_breaks_schema_not_parse():
    s = score_output(RECORD, answer(action_id="reboot_everything"), ACTIONS)
    assert s.json_valid and not s.schema_valid and not s.action_match


def test_extra_keys_break_schema():
    obj = json.loads(answer())
    obj["confidence"] = 0.9
    assert not schema_valid(obj, ACTIONS)


def test_garbage_scores_zero():
    s = score_output(RECORD, "I think the database is slow.", ACTIONS)
    assert not s.json_valid and not s.root_cause_hit and s.matched_groups == 0


def test_lenient_parse_skips_broken_objects():
    assert lenient_parse('{"a": } then {"b": 1}') == {"b": 1}


def test_summary_counts_and_percentiles():
    scores = [score_output(RECORD, answer(), ACTIONS, latency_s=t, output_tokens=10)
              for t in (1.0, 2.0, 3.0, 4.0)]
    scores.append(score_output(RECORD, "nope", ACTIONS, latency_s=5.0, output_tokens=10))
    s = summarise(scores)
    assert s["n"] == 5
    assert s["root_cause_hit"] == 4 and s["root_cause_hit_pct"] == 80.0
    assert s["latency_p50_s"] == 3.0
    assert s["latency_p95_s"] == 4.8
    assert s["output_tokens_per_s"] == round(50 / 15, 2)
    assert s["by_scenario"]["SC-001"]["n"] == 5
