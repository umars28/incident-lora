#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
ADAPTER="${ADAPTER:-runs/qwen05-r16/adapter}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
VLLM="${VLLM:-vllm}"
export VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"

exec "$VLLM" serve "$BASE_MODEL" \
  --served-model-name qwen2.5-0.5b-instruct \
  --enable-lora \
  --lora-modules "incident-lora=$ADAPTER" \
  --max-lora-rank 16 \
  --max-loras 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_MODEL_LEN" \
  --dtype "${DTYPE:-auto}" \
  --port "$PORT" \
  "$@"
