#!/usr/bin/env bash
set -euo pipefail

# Reproducible environment for Qwen3-Embedding-0.6B scripts.
#
# Usage:
#   bash setup_qwen_env.sh
#
# Optional variables:
#   VENV_DIR=.venv
#   PYTHON_BIN=python3
#   PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
#   WHEELHOUSE=/path/to/offline/wheels
#
# After setup:
#   source "${VENV_DIR:-.venv}/bin/activate"
#   python download_qwen_embedding.py
#   python embed_score.py

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

TORCH_SPEC="${TORCH_SPEC:-torch>=2.6,<3}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers>=4.51,<5}"
NUMPY_SPEC="${NUMPY_SPEC:-numpy>=1.26,<3}"
PANDAS_SPEC="${PANDAS_SPEC:-pandas==2.2.3}"

echo "==> Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(
        f"Python >= 3.10 is required; found {sys.version.split()[0]}"
    )
print(f"Python version OK: {sys.version.split()[0]}")
PY

echo "==> Creating virtual environment: ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading packaging tools"
python -m pip install --upgrade pip setuptools wheel

pip_install() {
    if [[ -n "${WHEELHOUSE:-}" ]]; then
        python -m pip install --no-index --find-links "${WHEELHOUSE}" "$@"
    else
        python -m pip install "$@"
    fi
}

echo "==> Installing PyTorch"
if [[ -n "${WHEELHOUSE:-}" ]]; then
    pip_install "${TORCH_SPEC}"
elif [[ -n "${PYTORCH_INDEX_URL:-}" ]]; then
    python -m pip install --index-url "${PYTORCH_INDEX_URL}" "${TORCH_SPEC}"
else
    pip_install "${TORCH_SPEC}"
fi

echo "==> Installing Python libraries used by the scripts"
pip_install     "${TRANSFORMERS_SPEC}"     "${NUMPY_SPEC}"     "${PANDAS_SPEC}"

echo "==> Verifying imports and versions"
python - <<'PY'
import numpy
import pandas
import torch
import transformers

print("numpy       :", numpy.__version__)
print("pandas      :", pandas.__version__)
print("torch       :", torch.__version__)
print("transformers:", transformers.__version__)
print("CUDA avail. :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA runtime:", torch.version.cuda)
    print("GPU         :", torch.cuda.get_device_name(0))
PY

echo "==> Saving exact installed versions"
python -m pip freeze > requirements.lock.txt

echo
echo "Environment ready."
echo "Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
echo
echo "Download the model once:"
echo "  python download_qwen_embedding.py"
echo
echo "Then generate embeddings:"
echo "  python embed_score.py"
echo
echo "Exact installed versions:"
echo "  requirements.lock.txt"
