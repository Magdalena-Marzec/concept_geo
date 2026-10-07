#!/bin/bash -l
#SBATCH --job-name=score_embed
#SBATCH --partition=gpu
#SBATCH --account=g104-2594
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta32G
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-embed-%j.out
#SBATCH --error=slurm-embed-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module use /apps/modulefiles/common/common
module load anaconda/23.3.1

source .venv/bin/activate

mkdir -p logs

JOB_ID="${SLURM_JOB_ID:-manual}"
LOG_FILE="logs/embed_${JOB_ID}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "SCORE / Qwen embedding run"
echo "Job ID: $JOB_ID"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Log: $LOG_FILE"
echo "============================================================"

echo
echo "=== SOFTWARE ==="
python --version

python - <<'PY'
import numpy
import pandas
import torch
import transformers

print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("transformers:", transformers.__version__)
PY

echo
echo "=== GPU ==="
nvidia-smi

python - <<'PY'
import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

print("GPU:", torch.cuda.get_device_name(0))
PY

echo
echo "=== INPUT CHECK ==="

python - <<'PY'
import pandas as pd

cols = [
    "concept",
    "definition",
    "paraphrase_1",
    "paraphrase_2",
    "paraphrase_3",
    "paraphrase_4",
    "paraphrase_5",
    "paraphrase_6",
    "paraphrase_7",
]

df = pd.read_csv("score_concepts.csv")

print("Rows:", len(df))
print("Embedding columns:", len(cols))
print("Expected embeddings:", len(df) * len(cols))
print("Missing values:", int(df[cols].isna().sum().sum()))

if len(df) != 214:
    raise RuntimeError(f"Expected 214 concepts, got {len(df)}")

if df[cols].isna().any().any():
    raise RuntimeError("Missing text found in embedding columns")
PY

test -f embed_score.py
test -f models/Qwen3-Embedding-0.6B/config.json
test -f models/Qwen3-Embedding-0.6B/model.safetensors

echo
echo "=== EMBEDDING START ==="
date

rm -f score_embeddings.npz
rm -f score_embeddings_metadata.csv

python -u embed_score.py

echo
echo "=== EMBEDDING FINISHED ==="
date

echo
echo "=== OUTPUT VALIDATION ==="

python - <<'PY'
import numpy as np
import pandas as pd

npz = np.load("score_embeddings.npz")

print("NPZ keys:", list(npz.keys()))

if "embeddings" not in npz:
    raise RuntimeError("NPZ does not contain 'embeddings'")

emb = npz["embeddings"]
meta = pd.read_csv("score_embeddings_metadata.csv")

print("Embedding shape:", emb.shape)
print("Embedding dtype:", emb.dtype)
print("Metadata rows:", len(meta))
print("Finite:", np.isfinite(emb).all())

norms = np.linalg.norm(emb, axis=1)

print("Norm min:", norms.min())
print("Norm max:", norms.max())
print("Norm mean:", norms.mean())

if emb.shape != (1926, 1024):
    raise RuntimeError(
        f"Unexpected embedding shape: {emb.shape}; expected (1926, 1024)"
    )

if len(meta) != 1926:
    raise RuntimeError(
        f"Unexpected metadata row count: {len(meta)}"
    )

if not np.isfinite(emb).all():
    raise RuntimeError("Embeddings contain NaN or Inf")

if not np.allclose(norms, 1.0, atol=1e-5):
    raise RuntimeError("Embeddings are not L2-normalized")

print("OUTPUT VALIDATION OK")
PY

echo
echo "=== OUTPUT FILES ==="
ls -lh score_embeddings.npz score_embeddings_metadata.csv

echo
echo "============================================================"
echo "EMBEDDING JOB COMPLETED SUCCESSFULLY"
echo "Date: $(date)"
echo "============================================================"
