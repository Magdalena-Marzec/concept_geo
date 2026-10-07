#!/bin/bash -l
#SBATCH --job-name=qwen_download
#SBATCH --partition=gpu
#SBATCH --account=g104-2594
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=qwen_download-%j.out
#SBATCH --error=qwen_download-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module use /apps/modulefiles/common/common
module load anaconda/23.3.1

source .venv/bin/activate

echo "=== HOST ==="
hostname

echo
echo "=== PYTHON ==="
which python
python --version

echo
echo "=== DOWNLOAD QWEN ==="
python -u download_qwen_embedding.py

echo
echo "=== CHECK MODEL FILES ==="
du -sh models/Qwen3-Embedding-0.6B
ls -lh models/Qwen3-Embedding-0.6B

echo
echo "=== LOCAL LOAD TEST ==="

python - <<'PY'
from transformers import AutoTokenizer, AutoModel

path = "./models/Qwen3-Embedding-0.6B"

tokenizer = AutoTokenizer.from_pretrained(
    path,
    local_files_only=True
)

model = AutoModel.from_pretrained(
    path,
    local_files_only=True
)

print("TOKENIZER OK")
print("MODEL OK")
print("model_type:", model.config.model_type)
print("hidden_size:", model.config.hidden_size)
PY

echo
echo "DOWNLOAD AND LOCAL LOAD TEST OK"
