#!/bin/bash -l
#SBATCH --job-name=mag_emb_down
#SBATCH --partition=gpu
#SBATCH --account=g107-2945
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=mag_emb_down-%j.out
#SBATCH --error=mag_emb_down-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module use /apps/modulefiles/common/common
module load anaconda/23.3.1
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
# Downloads require internet connectivity on the assigned node.
# For NV-Embed-v2 accept model terms / login on Hugging Face first.
python -u -m v100_next_models.download --models ${MODELS:-medembed pubmedbert sapbert nvembed nemotron_medical}
