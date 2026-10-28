#!/bin/bash -l
#SBATCH --job-name=qwen3_tatum
#SBATCH --partition=gpu
#SBATCH --account=g104-2594
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta32G
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=qwen3_tatum-%j.out
#SBATCH --error=qwen3_tatum-%j.err

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
exec > >(tee -a "logs/qwen3_tatum_${SLURM_JOB_ID}.log") 2>&1

echo "Tatum embeddings: $SLURM_JOB_ID, $(date), $(hostname)"
python tatum/prepare_inputs.py
python -u tatum/embed_tatum_qwen.py --models 0.6B 4B 8B
python - <<'PY'
import json, numpy as np, pandas as pd
from pathlib import Path
p = Path('embedding_results/tatum')
m = json.loads((p/'tatum_embedding_manifest.json').read_text())
ids = pd.read_csv(p/'tatum_embeddings_metadata.csv')
assert len(ids)==m['total_texts']
for slug, n in [('0_6b', 1024), ('4b',2560), ('8b',4096)]:
    with np.load(p/f'tatum_qwen3_{slug}.npz') as z:
        assert z['embeddings_native'].shape==(len(ids),n)
        assert z['embeddings_1024'].shape==(len(ids),1024)
        for key in ('embeddings_native','embeddings_1024'):
            x=z[key]
            assert np.isfinite(x).all()
            assert np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-5)
print('TATUM FINAL VALIDATION OK')
PY
