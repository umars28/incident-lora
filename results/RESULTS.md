## Overall

| model | test set | n | root cause hit | category | action | strict correct | valid JSON | valid schema | p50 s | p95 s | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B base | test_heldout | 36 | 4/36 (11.1%) | 6/36 (16.7%) | 3/36 (8.3%) | 0/36 (0.0%) | 36/36 (100.0%) | 12/36 (33.3%) | 1.6 | 4.761 | 64.88 |
| Qwen2.5-0.5B base | test_seen | 21 | 0/21 (0.0%) | 5/21 (23.8%) | 8/21 (38.1%) | 0/21 (0.0%) | 18/21 (85.7%) | 9/21 (42.9%) | 1.614 | 6.375 | 54.37 |
| Qwen2.5-0.5B + LoRA | test_heldout | 36 | 16/36 (44.4%) | 9/36 (25.0%) | 19/36 (52.8%) | 10/36 (27.8%) | 35/36 (97.2%) | 31/36 (86.1%) | 3.152 | 4.594 | 42.74 |
| Qwen2.5-0.5B + LoRA | test_seen | 21 | 19/21 (90.5%) | 21/21 (100.0%) | 21/21 (100.0%) | 19/21 (90.5%) | 21/21 (100.0%) | 21/21 (100.0%) | 2.996 | 3.765 | 42.02 |
| vLLM CPU fp32: base | test_heldout | 36 | 7/36 (19.4%) | 6/36 (16.7%) | 2/36 (5.6%) | 0/36 (0.0%) | 36/36 (100.0%) | 4/36 (11.1%) | 6.21 | 12.52 | 23.79 |
| vLLM CPU fp32: incident-lora | test_heldout | 36 | 18/36 (50.0%) | 8/36 (22.2%) | 20/36 (55.6%) | 13/36 (36.1%) | 35/36 (97.2%) | 33/36 (91.7%) | 17.315 | 21.721 | 8.43 |
| qwen3:8b via Ollama (reference) | test_heldout | 36 | 19/36 (52.8%) | 18/36 (50.0%) | 27/36 (75.0%) | 15/36 (41.7%) | 36/36 (100.0%) | 36/36 (100.0%) | 9.974 | 11.926 | 9.1 |
| qwen3:8b via Ollama (reference) | test_seen | 21 | 20/21 (95.2%) | 8/21 (38.1%) | 15/21 (71.4%) | 14/21 (66.7%) | 21/21 (100.0%) | 20/21 (95.2%) | 10.707 | 12.681 | 9.32 |
| oracle (ground truth) | test_heldout | 36 | 36/36 (100.0%) | 36/36 (100.0%) | 36/36 (100.0%) | 36/36 (100.0%) | 36/36 (100.0%) | 36/36 (100.0%) | 0.0 | 0.0 | 0.0 |
| oracle (ground truth) | test_seen | 21 | 19/21 (90.5%) | 21/21 (100.0%) | 21/21 (100.0%) | 19/21 (90.5%) | 21/21 (100.0%) | 21/21 (100.0%) | 0.0 | 0.0 | 0.0 |

## Per scenario: test_heldout

| scenario | Qwen2.5-0.5B base | Qwen2.5-0.5B + LoRA | vLLM CPU fp32: base | vLLM CPU fp32: incident-lora | qwen3:8b via Ollama (reference) | oracle (ground truth) |
|---|---|---|---|---|---|---|
| SC-015 | 0/4 rc, 0/4 cat, 0/4 act | 2/4 rc, 0/4 cat, 1/4 act | 2/4 rc, 1/4 cat, 0/4 act | 2/4 rc, 0/4 cat, 0/4 act | 4/4 rc, 0/4 cat, 4/4 act | 4/4 rc, 4/4 cat, 4/4 act |
| SC-020 | 1/4 rc, 0/4 cat, 0/4 act | 1/4 rc, 4/4 cat, 0/4 act | 1/4 rc, 0/4 cat, 0/4 act | 2/4 rc, 3/4 cat, 0/4 act | 3/4 rc, 3/4 cat, 2/4 act | 4/4 rc, 4/4 cat, 4/4 act |
| SC-021 | 2/4 rc, 3/4 cat, 0/4 act | 2/4 rc, 3/4 cat, 3/4 act | 2/4 rc, 2/4 cat, 0/4 act | 3/4 rc, 2/4 cat, 2/4 act | 1/4 rc, 4/4 cat, 3/4 act | 4/4 rc, 4/4 cat, 4/4 act |
| SC-022 | 1/4 rc, 0/4 cat, 0/4 act | 1/4 rc, 0/4 cat, 1/4 act | 2/4 rc, 0/4 cat, 0/4 act | 3/4 rc, 0/4 cat, 3/4 act | 4/4 rc, 0/4 cat, 4/4 act | 4/4 rc, 4/4 cat, 4/4 act |
| SC-023 | 0/4 rc, 0/4 cat, 0/4 act | 0/4 rc, 2/4 cat, 3/4 act | 0/4 rc, 0/4 cat, 0/4 act | 0/4 rc, 3/4 cat, 4/4 act | 0/4 rc, 4/4 cat, 4/4 act | 4/4 rc, 4/4 cat, 4/4 act |
| SC-024 | 0/4 rc, 3/4 cat, 0/4 act | 4/4 rc, 0/4 cat, 3/4 act | 0/4 rc, 3/4 cat, 0/4 act | 3/4 rc, 0/4 cat, 4/4 act | 2/4 rc, 4/4 cat, 4/4 act | 4/4 rc, 4/4 cat, 4/4 act |
| SC-028 | 0/4 rc, 0/4 cat, 0/4 act | 0/4 rc, 0/4 cat, 0/4 act | 0/4 rc, 0/4 cat, 0/4 act | 0/4 rc, 0/4 cat, 0/4 act | 1/4 rc, 3/4 cat, 4/4 act | 4/4 rc, 4/4 cat, 4/4 act |
| SC-029 | 0/4 rc, 0/4 cat, 3/4 act | 2/4 rc, 0/4 cat, 4/4 act | 0/4 rc, 0/4 cat, 2/4 act | 2/4 rc, 0/4 cat, 3/4 act | 1/4 rc, 0/4 cat, 1/4 act | 4/4 rc, 4/4 cat, 4/4 act |
| SC-030 | 0/4 rc, 0/4 cat, 0/4 act | 4/4 rc, 0/4 cat, 4/4 act | 0/4 rc, 0/4 cat, 0/4 act | 3/4 rc, 0/4 cat, 4/4 act | 3/4 rc, 0/4 cat, 1/4 act | 4/4 rc, 4/4 cat, 4/4 act |

## Per scenario: test_seen

| scenario | Qwen2.5-0.5B base | Qwen2.5-0.5B + LoRA | qwen3:8b via Ollama (reference) | oracle (ground truth) |
|---|---|---|---|---|
| SC-001 | 0/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 0/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 1/1 act |
| SC-002 | 0/1 rc, 1/1 cat, 1/1 act | 0/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 0/1 cat, 1/1 act | 0/1 rc, 1/1 cat, 1/1 act |
| SC-003 | 0/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 1/1 act |
| SC-004 | 0/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 0/1 cat, 0/1 act | 1/1 rc, 1/1 cat, 1/1 act |
| SC-005 | 0/1 rc, 0/1 cat, 0/1 act | 1/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 0/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 1/1 act |
| SC-006 | 0/1 rc, 0/1 cat, 0/1 act | 1/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 1/1 act |
| SC-007 | 0/1 rc, 0/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 0/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 1/1 act |
| SC-008 | 0/1 rc, 0/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 0/1 cat, 0/1 act | 1/1 rc, 1/1 cat, 1/1 act |
| SC-009 | 0/1 rc, 0/1 cat, 0/1 act | 1/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 0/1 cat, 0/1 act | 1/1 rc, 1/1 cat, 1/1 act |
| SC-010 | 0/1 rc, 0/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 1/1 act |
| SC-011 | 0/1 rc, 0/1 cat, 0/1 act | 1/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 0/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 1/1 act |
| SC-012 | 0/1 rc, 0/1 cat, 0/1 act | 1/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 0/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 1/1 act |
| SC-013 | 0/1 rc, 1/1 cat, 0/1 act | 1/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 1/1 act |
| SC-014 | 0/1 rc, 0/1 cat, 0/1 act | 1/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 1/1 act |
| SC-016 | 0/1 rc, 0/1 cat, 0/1 act | 1/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 0/1 act | 1/1 rc, 1/1 cat, 1/1 act |
| SC-017 | 0/1 rc, 0/1 cat, 0/1 act | 0/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 0/1 act | 0/1 rc, 1/1 cat, 1/1 act |
| SC-018 | 0/1 rc, 0/1 cat, 0/1 act | 1/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 0/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 1/1 act |
| SC-019 | 0/1 rc, 0/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 0/1 cat, 0/1 act | 1/1 rc, 1/1 cat, 1/1 act |
| SC-025 | 0/1 rc, 0/1 cat, 0/1 act | 1/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 0/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 1/1 act |
| SC-026 | 0/1 rc, 0/1 cat, 0/1 act | 1/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 1/1 act |
| SC-027 | 0/1 rc, 0/1 cat, 0/1 act | 1/1 rc, 1/1 cat, 1/1 act | 0/1 rc, 0/1 cat, 1/1 act | 1/1 rc, 1/1 cat, 1/1 act |

## Load tests

| server | model | concurrency | requests | ok | wall s | req/s | out tok/s | p50 s | p95 s |
|---|---|---|---|---|---|---|---|---|---|
| vllm float32 | incident-lora | 1 | 8 | 8 | 88.931 | 0.09 | 13.3 | 10.708 | 14.075 |
| vllm float32 | incident-lora | 4 | 16 | 16 | 121.921 | 0.1312 | 19.03 | 30.189 | 41.247 |
| vllm float32 | incident-lora | 8 | 16 | 16 | 103.636 | 0.1544 | 22.39 | 49.332 | 79.797 |
