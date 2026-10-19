#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
ADAPTER="${ADAPTER:-adapters/qwen2.5-0.5b-incident-r16}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
VLLM="${VLLM:-vllm}"

exec "$VLLM" serve "$BASE_MODEL" \
  --served-model-name qwen2.5-0.5b-instruct \
  --enable-lora \
  --lora-modules "incident-lora=$ADAPTER" \
  --max-lora-rank 16 \
  --max-loras 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --dtype "${DTYPE:-auto}" \
  --port "$PORT" \
  "$@"
