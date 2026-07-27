#!/bin/bash
#SBATCH --job-name=random_1d_4gpu
#SBATCH --output=random_1d_4gpu_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -e

cd "${SLURM_SUBMIT_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate data

nvidia-smi

srun python -u src/random_search_cnn_opt_1d.py \
  --trials 500 \
  --seeds 0 \
  --num-gpus 4 \
  --epochs 25 \
  --batch-size 256 \
  --auto-summarize
