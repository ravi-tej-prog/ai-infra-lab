#!/usr/bin/env bash
# Send one timed chat request to an OpenAI-compatible server.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi
cd "$ROOT"
exec "$PYTHON" -m qwen_platform infer "$@"
