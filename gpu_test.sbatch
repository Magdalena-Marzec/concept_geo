#!/bin/bash -l
#SBATCH --job-name=gpu_test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=gpu_test-%j.out
#SBATCH --error=gpu_test-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

echo "=== HOST ==="
hostname
echo

echo "=== LOAD PYTHON ==="
module use /apps/modulefiles/common/common
module load anaconda/23.3.1

echo "python before venv:"
which python
python --version

source .venv/bin/activate

echo
echo "python after venv:"
which python
python --version

echo
echo "=== NVIDIA-SMI ==="
nvidia-smi

echo
echo "=== PYTORCH GPU TEST ==="

python - <<'PY'
import sys
import torch

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access CUDA on this compute node.")

print("GPU name:", torch.cuda.get_device_name(0))

# Faktyczna operacja na GPU
x = torch.randn(1000, 1000, device="cuda")
y = x @ x

print("Tensor device:", y.device)
print("Result finite:", torch.isfinite(y).all().item())
print("Result shape:", tuple(y.shape))

print()
print("GPU TEST OK")
PY
