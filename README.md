# incident-lora

A LoRA adapter for Qwen2.5-0.5B-Instruct, trained with Hugging Face PEFT to read a frozen production incident and answer with a JSON diagnosis: the root cause, a failure category and one remediation action. It is measured against its own base model and against qwen3:8b on incidents it never saw during training, and it is served with vLLM as an OpenAI-compatible endpoint that hosts the base model and the adapter side by side.

The organising rule comes from [llm-platform](../llm-platform): every claim in this README is a number produced by a command you can run. The commands are at the bottom, and every number below can be found in a file under `results/`.

## At a glance

| | |
|---|---|
| **Task** | Read a production incident (alert, logs, metrics, Kubernetes objects, runbooks) and return a JSON diagnosis: root cause, failure category, remediation action. |
| **Model** | Qwen2.5-0.5B-Instruct + LoRA (PEFT). It trains 1.75% of the parameters, and the adapter is 35 MB. Training took about 72 minutes on an Apple M5 laptop. |
| **Headline result** | On 9 incident scenarios never seen in training (36 prompts), LoRA raises root cause hits from **11.1% → 44.4%**, correct action from **8.3% → 52.8%**, and schema-valid answers from **33.3% → 86.1%**. |
| **Honest comparison** | qwen3:8b (16× larger) still wins on every held-out metric (41.7% vs 27.8% strict correct), at about 3× the latency per answer. The adapter's near-perfect seen-scenario score is reported as memorisation, not as the headline. |
| **No leakage** | Whole scenarios are held out by a mechanical rule. The build checks for 0 identical prompts across splits and at most 1.54% shared 5-grams with training targets. The checks are recorded in `results/dataset_manifest.json`. |
| **Serving** | vLLM serves the base model and the adapter side by side on one OpenAI-compatible endpoint. Results are load-tested at concurrency 1/4/8. A small shim exposes it on the Anthropic Messages API for [llm-gateway](https://github.com/umars28/llm-platform). |
| **Found along the way** | A vLLM prefix-caching bug turned every answer after the first into prose (0/36 valid JSON). It was reproduced in both dtypes, isolated, and the failing runs are kept in `results/vllm-prefix-cache-bug/`. Two source scenarios had ground truth that fails its own scorer. |
| **Stack** | Python, PyTorch (MPS), Hugging Face Transformers + PEFT, vLLM, uv, pytest |

Every number above comes from the tables below and the JSON files in `results/`.

## Results

Held-out set: 36 prompts built from 9 incident scenarios that never appear in training, 4 prompts per scenario (the original plus 3 variants).

| model | root cause hit | category | action | strict correct | valid JSON | valid schema | p50 latency |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B base | 4/36 (11.1%) | 6/36 (16.7%) | 3/36 (8.3%) | 0/36 (0.0%) | 36/36 (100.0%) | 12/36 (33.3%) | 1.6 s |
| Qwen2.5-0.5B + LoRA | 16/36 (44.4%) | 9/36 (25.0%) | 19/36 (52.8%) | 10/36 (27.8%) | 35/36 (97.2%) | 31/36 (86.1%) | 3.152 s |
| qwen3:8b via Ollama | 19/36 (52.8%) | 18/36 (50.0%) | 27/36 (75.0%) | 15/36 (41.7%) | 36/36 (100.0%) | 36/36 (100.0%) | 9.974 s |
| ground truth (ceiling) | 36/36 | 36/36 | 36/36 | 36/36 | 36/36 | 36/36 | |

Seen-scenario set: 21 new variants of the 21 training scenarios, generated with a seed the training set never used.

| model | root cause hit | category | action | strict correct | valid schema |
|---|---|---|---|---|---|
| Qwen2.5-0.5B base | 0/21 (0.0%) | 5/21 (23.8%) | 8/21 (38.1%) | 0/21 (0.0%) | 9/21 (42.9%) |
| Qwen2.5-0.5B + LoRA | 19/21 (90.5%) | 21/21 (100.0%) | 21/21 (100.0%) | 19/21 (90.5%) | 21/21 (100.0%) |
| qwen3:8b via Ollama | 20/21 (95.2%) | 8/21 (38.1%) | 15/21 (71.4%) | 14/21 (66.7%) | 20/21 (95.2%) |
| ground truth (ceiling) | 19/21 (90.5%) | 21/21 | 21/21 | 19/21 (90.5%) | 21/21 |

Training cost, from `results/training/qwen05-r16-train_meta.json`:

| | |
|---|---|
| trainable parameters | 8,798,208 of 502,830,976 (1.7497%) |
| adapter on disk | 35,237,104 bytes (fp32 safetensors) |
| training examples | 504, 827,019 tokens per epoch |
| optimizer steps | 126 (2 epochs, effective batch 8) |
| wall-clock time | 4330.9 s on Apple M5 (MPS) |
| validation loss | 2.464054584503174 before training, 0.007666604948185739 after epoch 1, 0.001915912737049872 after epoch 2 |

## What the numbers say

The adapter helps on incidents it has never seen, and the gap between the two test sets shows how much of what it learned is memorisation.

On the held-out scenarios, LoRA takes the 0.5B model from 11.1% to 44.4% on root cause, from 8.3% to 52.8% on choosing an acceptable action, and from 33.3% to 86.1% on producing an answer that fits the schema. The base model emits parseable JSON for all 36 held-out prompts but fills it with categories and action ids that do not exist; teaching the contract is the cheapest win LoRA delivers here. qwen3:8b, sixteen times larger and run through Ollama, still beats the adapter on every held-out metric, and it costs about three times the latency per answer on this machine.

On the seen scenarios the adapter scores 19 of 21 on root cause, which is exactly the ceiling: the two misses are the two scenarios whose own ground truth text fails its keyword groups (see below). Category and action are perfect. That is what memorisation of 21 incident templates looks like, and it is why the held-out set and not this one is the headline. Validation loss fell to 0.001915912737049872 for the same reason; the validation set is drawn from training scenarios, so it measures fit, not generalisation.

Category accuracy on held-out incidents is the weakest result: 9 of 36 for the adapter. Five of the 21 training scenarios are `resource-exhaustion`, and the adapter labels most unfamiliar incidents that way. The per-scenario table in `results/RESULTS.md` shows it never once labels SC-022 (a health check path change) as `config-error` or SC-029 (a Kafka rebalance loop) as `dependency-failure`. qwen3:8b also gets 0 of 4 on SC-015, SC-022, SC-029 and SC-030 for category, which suggests part of the problem is the taxonomy itself rather than the model.

The root cause metric deserves a caveat. It is the keyword scorer from llm-platform's ops-copilot, unchanged: a diagnosis counts when it contains one phrase from every `must_include` group. It is cheap, deterministic and never needs its own evaluation, and it cannot tell a correct explanation from a fluent one that quotes the right evidence. Reading the adapter's held-out outputs, some of its 16 hits are sound (the original SC-030 answer names the 24Gi limit raised against a 2Gi request) while others reach the right keywords through a muddled story (the original SC-029 answer). The strict column, which also requires an acceptable action and no forbidden claim, is the more honest of the two numbers.

## The task

The input is one incident from `llm-platform/benchmarks/incidents`, rendered as text: the alert, the service graph, deploys, logs, metric series, Kubernetes objects and the scenario's runbooks. A training example, prompt plus answer, has a median of 1,607 tokens and a maximum of 2,133 (counted with the Qwen tokenizer by `ilora stats`). The system prompt lists the ten categories and the 25 remediation actions from the shared catalogue, so the base model and the adapter see exactly the same instructions.

The expected output is one JSON object:

```json
{"root_cause": "...", "category": "resource-exhaustion", "action_id": "rollback_deploy"}
```

`root_cause` is trained on the scenario's ground truth text. `category` is the scenario's family. `action_id` is trained on the first expected action, and scored as correct if it is any of the expected actions, the same rule ops-copilot uses. The `root_cause` key comes first so the model writes its reasoning before it commits to a label.

## Data, split and why it cannot leak

Thirty scenarios are far too few to train on, so `ilora data` generates variants of each one. A variant renames every non-infrastructure service consistently across the alert, logs, metrics, Kubernetes objects, runbooks and the ground truth (including the keyword groups, so the label still matches the prompt). It shifts version numbers by a consistent major delta so upgrades still look like upgrades, renames node identifiers such as `ip-10-4-7-12`, replaces commit hashes, moves the alert to a random time of year, scales uncited metric series by up to 5% in either direction, injects benign log lines, sometimes adds a harmless deploy to an unrelated service, and shuffles the order of services, deploys, log streams, metric series and Kubernetes objects. Infrastructure names such as `postgres-primary` are kept because the ground truth argues about Postgres by name. Any metric series whose values are quoted anywhere else in the scenario is left untouched, so "97 of 400 connections" in the ground truth stays true.

The split holds out whole scenarios. The rule is mechanical so it cannot be tuned after the fact: for every category with at least two scenarios, the highest-numbered one is held out. That selects SC-015, SC-020, SC-021, SC-022, SC-023, SC-024, SC-028, SC-029 and SC-030, one per category, and leaves 21 scenarios for training. `misleading-signal` has a single scenario (SC-026) and stays in training, because a held-out category the model has never seen a label for measures nothing.

| set | scenarios | prompts | purpose |
|---|---|---|---|
| train | 21 | 504 (24 per scenario, original included) | fit the adapter |
| val | 21 | 21 | validation loss during training |
| test_seen | 21 | 21 | new variants of training scenarios, to measure memorisation |
| test_heldout | 9 | 36 (4 per scenario, original included) | generalisation, the headline |

Leakage is checked when the data is built and written to `results/dataset_manifest.json`. No held-out scenario id appears in training. No test or validation prompt is byte-identical to a training prompt (0 of 36, 0 of 21, 0 of 21). The held-out ground truth texts share at most 1.54% of their word 5-grams with any training target (0.17% on average), and no runbook id appears in both a held-out and a training scenario. Variants are seeded from a hash of the base seed, scenario id, split name and index, so the seen-scenario test variants are produced by seeds the training set never used.

The scenarios are read from the sibling `llm-platform` checkout rather than copied, because llm-platform merged its repositories precisely to stop hand-copied incident fixtures drifting apart. The manifest records the sha256 of every scenario file so a changed corpus is visible. Set `INCIDENT_SCENARIOS_DIR` to point elsewhere.

## Training

`ilora train` is a plain PyTorch loop over `transformers` and `peft`, 212 lines, with no Trainer. LoRA rank 16, alpha 32, dropout 0.05, applied to all seven projection matrices in every block (q, k, v, o, gate, up, down). AdamW at a peak learning rate of 2e-4 with 5% linear warmup and cosine decay, gradient clipping at 1.0, batch size 1 with 8 steps of gradient accumulation, two epochs, bf16 base weights with fp32 adapter weights, gradient checkpointing on.

Loss is computed on the answer tokens only. The median prompt is 1,479 tokens and the median answer 127, so a loss over the whole sequence would mostly train the model to predict incident logs. The loop also asks the model for only the last few logits (`logits_to_keep`) instead of a 1,607 by 151,936 matrix. `ilora bench-step` measures one LoRA forward and backward pass at the median training length: with full logits the Metal driver held 7,297,794,048 bytes, with answer-only logits 3,394,977,792, at similar speed (2.338 to 2.403 s against 2.16 to 2.206 s per step, `results/bench/`). Memory was the constraint that mattered here. Docker Desktop and other workloads held most of the 24 GB during training, swap was nearly full, and the 4330.9 s wall-clock time over 1,008 forward and backward passes reflects that; treat it as an upper bound for this hardware.

The adapter is 35 MB because PEFT saves it in fp32; it is not committed. `ilora train` reproduces it, and `serve/` expects it at `runs/qwen05-r16/adapter`.

## Serving with vLLM

`serve/vllm-serve.sh` starts one OpenAI-compatible `vllm serve` process that hosts the base model under the name `qwen2.5-0.5b-instruct` and the adapter under `incident-lora` (`--enable-lora --lora-modules incident-lora=...`). A client picks the model per request, so the base and the fine-tuned model are compared on the same server and the same batch scheduler, and more adapters can be added without loading another copy of the weights. `ilora eval --backend openai` scores whatever answers on `/v1/chat/completions`, `ilora ask` sends one incident, and `ilora loadtest` drives a fixed number of requests at a given concurrency and reports throughput and p50/p95 latency.

It was run on this Mac. vLLM 0.11.0 builds from source into a separate venv on macOS arm64 (`serve/install-vllm-macos.sh`) and selects its CPU backend; it needed `transformers` pinned below 5 and `VLLM_HOST_IP=127.0.0.1`, because the torch distributed rendezvous otherwise tried to reach the hostname a VPN client had mapped. `serve/vllm-serve-macos-cpu.sh` wraps those settings. These are CPU numbers on a laptop under memory pressure, not GPU numbers, and the scripts in `serve/` (including `compose.gpu.yaml` for a single NVIDIA GPU) are untested on a GPU.

Held-out accuracy through vLLM (CPU, float32, prefix caching off):

| model | root cause hit | category | action | strict correct | valid JSON | valid schema | p50 latency | p95 latency |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b-instruct | 7/36 (19.4%) | 6/36 (16.7%) | 2/36 (5.6%) | 0/36 (0.0%) | 36/36 (100.0%) | 4/36 (11.1%) | 6.21 s | 12.52 s |
| incident-lora | 18/36 (50.0%) | 8/36 (22.2%) | 20/36 (55.6%) | 13/36 (36.1%) | 35/36 (97.2%) | 33/36 (91.7%) | 17.315 s | 21.721 s |

The adapter's improvement survives the move from PyTorch on MPS in bf16 to vLLM on CPU in float32. The exact numbers differ from the first table because greedy decoding diverges as soon as two backends disagree on one token, and that disagreement is expected across dtypes and devices; the size and direction of the gain are what carry over.

Load test against `incident-lora`, cycling through the held-out prompts in order (11,866 prompt tokens over the 8 single-stream requests, up to 384 output tokens each):

| concurrency | requests | ok | wall time | requests/s | output tokens/s | p50 latency | p95 latency |
|---|---|---|---|---|---|---|---|
| 1 | 8 | 8 | 88.931 s | 0.09 | 13.3 | 10.708 s | 14.075 s |
| 4 | 16 | 16 | 121.921 s | 0.1312 | 19.03 | 30.189 s | 41.247 s |
| 8 | 16 | 16 | 103.636 s | 0.1544 | 22.39 | 49.332 s | 79.797 s |

Batching does what it should: going from one request at a time to eight raises output throughput from 13.3 to 22.39 tokens per second, and each request waits longer for it. On a CPU the gain is small because prefill of a 1,500-token prompt dominates. No GPU number exists in this repository; anything faster than the table above would be a guess.

The run found one real bug, in vLLM rather than here. With automatic prefix caching on (vLLM's default), the first request after start-up was correct and later requests that shared the long system prompt came back as unstructured prose: 0 of 36 valid JSON for the adapter and 0 of 36 for the base model in float32, and 0 of 36 and 1 of 36 in bf16, against 35 and 36 of 36 from the same models under transformers. Turning prefix caching off fixed it in both dtypes (8 of 8 valid JSON on a bf16 spot check). Every request in this workload shares a 559-token system prompt (as counted by the Qwen tokenizer), which is the exact case prefix caching exists for, so on a GPU it should be switched back on and re-verified rather than assumed broken there. The failing runs are kept in `results/vllm-prefix-cache-bug/` and the spot check in `results/vllm-dtype-check/`.

## Behind llm-gateway

llm-platform's [llm-gateway](../llm-platform/apps/llm-gateway) forwards to providers on the Anthropic Messages protocol: it posts to `{base_url}/v1/messages` and reads `usage.input_tokens` and `usage.output_tokens` back for cost settlement. vLLM 0.11 serves only the OpenAI protocol, so the adapter cannot be listed as a provider directly. `ilora shim` is a small translator that accepts `/v1/messages`, forwards to `/v1/chat/completions`, and maps the reply and its token usage back. With it running next to vLLM (`serve/compose.gpu.yaml` starts both), the gateway needs only a config entry:

```yaml
providers:
  - name: local-vllm
    base_url: http://messages-shim:8090
    api_key_env: LOCAL_VLLM_KEY
    priority: 30
    failure_threshold: 5
    probe_after_s: 30
    models:
      incident-lora: incident-lora
      qwen2.5-0.5b-instruct: qwen2.5-0.5b-instruct
```

Two things in the gateway would need a decision before that is useful, and neither is changed here. Its pricing table falls back to Opus prices for any model it does not know, so every self-hosted request would be charged against a tenant's budget as if it were `claude-opus-5`; a zero or amortised price for these two model names belongs in `pricing.py`. And the gateway fails over by model name, so a request for `incident-lora` can only ever go to this provider; there is no second provider to fail over to, which is the right behaviour for a fine-tuned model but means its breaker opening is a real outage for that model.

The shim is tested end to end over a real socket against a mocked upstream (`tests/test_serving.py`), and it was run once against the live vLLM CPU server: an Anthropic-format request for `incident-lora` on SC-030 returned HTTP 200 in 9.954 s with `usage` of 1,454 input and 119 output tokens and `stop_reason` `end_turn` (`results/shim-live-vllm-cpu.json`). The gateway itself was not started.

## Problems found in the source corpus

Two llm-platform scenarios have ground truth text that fails its own keyword groups, which is why the ceiling on the seen-scenario set is 19 of 21 rather than 21. SC-002 says the cache "grew" while the group expects "grows" or "unbounded" among others, and SC-017's root cause never uses any phrase from its `["config", "configmap", "change", "shared", ...]` group. Neither is in the held-out set, so the headline ceiling is 36 of 36, but ops-copilot's own 96.7% is scored against the same groups.

## Running it

Python 3.11 to 3.13 with `uv`. Tests run offline against fixture scenarios with no model download:

```
uv sync --extra train
uv run --group dev pytest
```

Reproducing the results, in order:

```
uv run ilora data
uv run ilora stats
uv run ilora train --out runs/qwen05-r16
uv run ilora eval --backend oracle --set test_heldout --label "oracle (ground truth)" --out results/eval-oracle-test_heldout.json
uv run ilora eval --backend hf --model Qwen/Qwen2.5-0.5B-Instruct --set test_heldout --label "Qwen2.5-0.5B base" --out results/eval-base05-test_heldout.json
uv run ilora eval --backend hf --model Qwen/Qwen2.5-0.5B-Instruct --adapter runs/qwen05-r16/adapter --set test_heldout --label "Qwen2.5-0.5B + LoRA" --out results/eval-lora05-test_heldout.json
uv run ilora eval --backend ollama --model qwen3:8b --set test_heldout --label "qwen3:8b via Ollama (reference)" --out results/eval-ollama-qwen3-8b-test_heldout.json
uv run ilora report
```

Repeat the `eval` lines with `--set test_seen` for the second table. The Ollama line needs `ollama pull qwen3:8b` and a running Ollama server.

Serving, on macOS:

```
./serve/install-vllm-macos.sh
./serve/vllm-serve-macos-cpu.sh
uv run ilora ask --model incident-lora
uv run ilora eval --backend openai --model incident-lora --label "vLLM CPU fp32: incident-lora" --out results/eval-vllm-cpu-fp32-lora-test_heldout.json
uv run ilora loadtest --model incident-lora --concurrency 4 --requests 16 --dtype float32 --out results/loadtest-vllm-cpu-fp32-c4.json
uv run ilora shim --upstream http://127.0.0.1:8000 --port 8090
uv run ilora bench-step --out results/bench/step-answer-logits.json
```

On a single NVIDIA GPU, `docker compose -f serve/compose.gpu.yaml up` starts vLLM and the shim together, and the same `ask`, `eval` and `loadtest` commands apply.

## Layout

```
src/incident_lora/
  variants.py       systematic variant generation
  split.py          the held-out rule
  dataset.py        jsonl writer and leak checks
  render.py         incident to prompt, ground truth to target
  train.py          the LoRA loop
  backends.py       transformers, Ollama and OpenAI-compatible inference
  scoring.py        JSON validity, schema, keyword root cause, action
  loadtest.py       concurrent load against a /v1/chat/completions server
  bench.py          one timed LoRA training step
  messages_shim.py  Anthropic Messages to OpenAI chat translation
serve/              vLLM launch scripts, GPU compose file
results/            every number in this README
```

## Licence

MIT.
