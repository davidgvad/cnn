#!/bin/bash
#SBATCH --job-name=cnn_paper
#SBATCH --output=cnn_paper_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

# Submit from the repo root with the Python environment active.
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-4}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

srun "${CNN_PYTHON:-python}" -u reproduction/reproduce.py "$@"
