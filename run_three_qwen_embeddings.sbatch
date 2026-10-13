#!/bin/bash -l
#SBATCH --job-name=qwen3_scale
#SBATCH --partition=gpu
#SBATCH --account=g104-2594
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta32G
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=qwen3_scale-%j.out
#SBATCH --error=qwen3_scale-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module use /apps/modulefiles/common/common
module load anaconda/23.3.1
source .venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs
LOG_FILE="logs/qwen3_scale_${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "Three-model Qwen3 embedding experiment"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "============================================================"

python --version
python - <<'PY'
import numpy, pandas, torch, transformers
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo
nvidia-smi

echo
echo "=== Input files ==="
ls -lh score_concepts_completed.csv natural_language_words.csv external_domain_definitions.csv

echo
echo "=== Local models ==="
du -sh \
  models/Qwen3-Embedding-0.6B \
  models/Qwen3-Embedding-4B \
  models/Qwen3-Embedding-8B

echo
echo "=== Embedding all three models ==="
python -u embed_three_qwen.py --models 0.6B 4B 8B

echo
echo "=== Output files ==="
ls -lh embedding_results/

echo
echo "=== Final validation ==="
python - <<'PY'
import json
import numpy as np
import pandas as pd
from pathlib import Path

out = Path("embedding_results")
manifest = json.loads((out / "embedding_manifest.json").read_text())

score_meta = pd.read_csv(out / "score_embeddings_metadata.csv")
natural_meta = pd.read_csv(out / "natural_language_embeddings_metadata.csv")
external_meta = pd.read_csv(out / "external_domain_embeddings_metadata.csv")

expected_score = manifest["score_texts"]
expected_natural = manifest["natural_words"]
expected_external = manifest["external_texts"]

assert len(score_meta) == expected_score
assert len(natural_meta) == expected_natural
assert len(external_meta) == expected_external

for slug, dim in [("0_6b", 1024), ("4b", 2560), ("8b", 4096)]:
    s = np.load(out / f"score_qwen3_{slug}.npz")
    n = np.load(out / f"natural_qwen3_{slug}.npz")
    e = np.load(out / f"external_qwen3_{slug}.npz")

    assert s["embeddings_native"].shape == (expected_score, dim)
    assert s["embeddings_1024"].shape == (expected_score, 1024)

    assert n["embeddings_native"].shape == (expected_natural, dim)
    assert n["embeddings_1024"].shape == (expected_natural, 1024)

    assert e["embeddings_native"].shape == (expected_external, dim)
    assert e["embeddings_1024"].shape == (expected_external, 1024)

    assert np.isfinite(s["embeddings_native"]).all()
    assert np.isfinite(n["embeddings_native"]).all()
    assert np.isfinite(e["embeddings_native"]).all()

print("FINAL VALIDATION OK")
print("SCORE texts:", expected_score)
print("Natural words:", expected_natural)
print("External texts:", expected_external)
PY

echo
echo "============================================================"
echo "THREE-MODEL EMBEDDING JOB COMPLETED SUCCESSFULLY"
echo "Date: $(date)"
echo "Log: $LOG_FILE"
echo "============================================================"
