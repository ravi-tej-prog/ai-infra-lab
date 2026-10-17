#!/usr/bin/env bash
# Print the vLLM serve command. This script does not start vLLM.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi
cd "$ROOT"
exec "$PYTHON" -m qwen_platform render-serve "$@"
