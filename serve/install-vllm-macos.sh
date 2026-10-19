#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p .vllm-env
uv venv -p 3.12 .vllm-env/.venv
VIRTUAL_ENV="$PWD/.vllm-env/.venv" uv pip install "vllm==0.11.0"
.vllm-env/.venv/bin/python -c "import vllm; from vllm.platforms import current_platform as p; print(vllm.__version__, type(p).__name__)"
