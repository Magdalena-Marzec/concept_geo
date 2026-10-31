#!/usr/bin/env bash
# macOS Apple Silicon only; creates a separate venv from the CUDA/Rysy one.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || {
  echo "ERROR: setup_mac.sh requires macOS on Apple Silicon (arm64)." >&2; exit 1;
}
if command -v python3.12 >/dev/null 2>&1; then
  PYTHON=python3.12
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON=python3.11
elif command -v brew >/dev/null 2>&1; then
  echo "Installing Python 3.12 through Homebrew..."
  brew install python@3.12
  PYTHON="$(brew --prefix python@3.12)/bin/python3.12"
else
  echo "ERROR: Need Python 3.11/3.12 (recommended: brew install python@3.12)." >&2
  exit 1
fi
if [[ ! -f .venv-mac/bin/python ]]; then
  "$PYTHON" -m venv .venv-mac
fi
"$ROOT/.venv-mac/bin/python" -m pip install --upgrade pip
"$ROOT/.venv-mac/bin/python" -m pip install -r requirements-mac.txt
"$ROOT/.venv-mac/bin/python" -m mac.embed_layers --check-inputs
"$ROOT/.venv-mac/bin/python" -m unittest discover -s mac/tests -v
echo "Environment ready. Run: bash run_mac.sh 3.8 smoke"
