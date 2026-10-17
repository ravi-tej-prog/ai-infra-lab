#!/usr/bin/env bash
# Run one closed-loop benchmark level, or write a not-run record if the server is down.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi
cd "$ROOT"
exec "$PYTHON" -m qwen_platform bench "$@"
