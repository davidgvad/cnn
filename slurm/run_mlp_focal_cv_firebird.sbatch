#!/bin/bash
#SBATCH --job-name=mlp_focal_cv
#SBATCH --output=/home/gvadzabd-laf/slurm_outputs/mlp_focal_cv_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=l40s:4
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gvadzabd@lafayette.edu
#SBATCH --chdir=/home/gvadzabd-laf/cnn

set -eo pipefail

echo "======================================================"
echo "MLP focal sweep started on $(hostname) at $(date)"
echo "Working directory: $(pwd)"
echo "Slurm job ID: ${SLURM_JOB_ID}"
echo "======================================================"

source /home/gvadzabd-laf/miniconda3/etc/profile.d/conda.sh
conda activate data

export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/cnn_mlp_pycache"
mkdir -p "${PYTHONPYCACHEPREFIX}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
which python
python --version
srun --ntasks=1 nvidia-smi -L

srun --ntasks=1 python -u src/tune_mlp_focal_cv_4gpu.py \
  --gpus 0 1 2 3

echo "======================================================"
echo "MLP focal sweep finished at $(date)"
echo "======================================================"
