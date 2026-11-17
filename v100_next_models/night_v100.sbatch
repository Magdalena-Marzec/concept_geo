#!/bin/bash -l
#SBATCH --job-name=mag_emb_night
#SBATCH --partition=gpu
#SBATCH --account=g107-2945
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta32G
#SBATCH --mem=48G
#SBATCH --time=18:00:00
#SBATCH --output=mag_emb_night-%j.out
#SBATCH --error=mag_emb_night-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module use /apps/modulefiles/common/common
module load anaconda/23.3.1
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p logs
exec > >(tee -a "logs/mag_emb_night_${SLURM_JOB_ID}.log") 2>&1
python -m v100_next_models.encode --dry-run
nvidia-smi
fail=0
# A fresh process per model prevents a failure from destroying earlier results.
# --require-smoke protects against untested architectures/checkpoints.
for model in ${MODELS:-medembed pubmedbert sapbert nvembed nemotron_medical}; do
  echo "======= FULL ${model} $(date) ======="
  if python -u -m v100_next_models.encode --model "$model" --full --require-smoke; then
    echo "SUCCESS: $model"
  else
    echo "FAILED: $model; other models will continue"
    fail=1
  fi
done
exit "$fail"
