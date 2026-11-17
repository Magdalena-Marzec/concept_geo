#!/bin/bash -l
#SBATCH --job-name=mag_medemb_probe
#SBATCH --partition=gpu
#SBATCH --account=g107-2945
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta32G
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=mag_medemb_probe-%j.out
#SBATCH --error=mag_medemb_probe-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module use /apps/modulefiles/common/common
module load anaconda/23.3.1
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi
# Do not include MedEmb into night_v100.sbatch before this passes, and
# verify text-only forward + EOS pooling against the authors' checkpoint.
python -u -m v100_next_models.encode --model "${MODEL:-medemb2b}" --smoke --allow-experimental
