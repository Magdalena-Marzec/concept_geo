#!/bin/bash -l
#SBATCH --job-name=qwen3_download
#SBATCH --partition=gpu
#SBATCH --account=g104-2594
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=qwen3_download-%j.out
#SBATCH --error=qwen3_download-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module use /apps/modulefiles/common/common
module load anaconda/23.3.1
source .venv/bin/activate

echo "Host: $(hostname)"
echo "Date: $(date)"
python --version

echo
echo "=== Qwen3-Embedding-4B ==="
python -u download_qwen3_embedding_4b.py

echo
echo "=== Qwen3-Embedding-8B ==="
python -u download_qwen3_embedding_8b.py

echo
echo "=== Downloaded model directories ==="
du -sh models/Qwen3-Embedding-4B models/Qwen3-Embedding-8B

echo
echo "DOWNLOAD JOB OK"
