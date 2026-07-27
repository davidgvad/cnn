#!/bin/bash
#SBATCH --job-name=cnn_opt_1d_4gpu
#SBATCH --output=cnn_opt_1d_4gpu_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -e

cd "${SLURM_SUBMIT_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate data

nvidia-smi

srun python -u src/cnn_opt_1d_4gpu.py \
  --num-gpus 4 \
  --epochs 25 \
  --batch-size 256 \
  --run-name opt1d_4gpu
