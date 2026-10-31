#!/usr/bin/env bash
# Usage: bash run_mac.sh 3.8 [check|smoke|score|tatum|controls|all] [--offline|--force|--max-length ...]
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
MODEL="${1:-3.8}"
TASK="${2:-all}"
if (( $# >= 2 )); then shift 2; elif (( $# == 1 )); then shift; fi
[[ "$MODEL" == 3.8 || "$MODEL" == 3.5 ]] || { echo "Model: 3.8 or 3.5" >&2; exit 2; }
PYTHON="$ROOT/.venv-mac/bin/python"
[[ -x "$PYTHON" ]] || { echo "Run bash setup_mac.sh first" >&2; exit 1; }
case "$TASK" in
  check) exec "$PYTHON" -m mac.embed_layers --check-inputs --model "$MODEL" --datasets score tatum natural external "$@" ;;
  smoke) exec "$PYTHON" -u -m mac.embed_layers --model "$MODEL" --datasets score tatum --limit 2 "$@" ;;
  score|tatum|natural|external) exec "$PYTHON" -u -m mac.embed_layers --model "$MODEL" --datasets "$TASK" "$@" ;;
  controls) exec "$PYTHON" -u -m mac.embed_layers --model "$MODEL" --datasets natural external "$@" ;;
  all) exec "$PYTHON" -u -m mac.embed_layers --model "$MODEL" --datasets score tatum natural external "$@" ;;
  *) echo "Task: check, smoke, score, tatum, controls, all" >&2; exit 2 ;;
esac
