#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export VLLM="${VLLM:-.vllm-env/.venv/bin/vllm}"
export DTYPE="${DTYPE:-float32}"
export VLLM_CPU_KVCACHE_SPACE="${VLLM_CPU_KVCACHE_SPACE:-4}"
exec ./serve/vllm-serve.sh --no-enable-prefix-caching "$@"
