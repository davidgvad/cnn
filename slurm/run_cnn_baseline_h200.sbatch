#!/bin/bash
#SBATCH --job-name=cnn_baseline_h200
#SBATCH --output=/home/gvadzabd-laf/slurm_outputs/cnn_baseline_h200_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=h200:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8gb
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gvadzabd@lafayette.edu
#SBATCH --chdir=/home/gvadzabd-laf/cnn

echo "======================================================"
echo "Job started on $(hostname) at $(date)"
echo "Working Directory: $(pwd)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "======================================================"

# Optional: keep Python bytecode cache off $HOME (friendlier on many HPC systems)
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/pycache"
mkdir -p "${PYTHONPYCACHEPREFIX}"

# --- Conda Initialization ---
source ~/miniconda3/etc/profile.d/conda.sh

# Activate your Conda environment (CHANGE if needed)
conda activate data

echo "======================================================"
echo "Python: $(which python)"
python --version
echo "======================================================"

echo "GPU visibility:"
nvidia-smi
echo "======================================================"

# --- Run the baseline CNN (NO CTGAN / NO undersampling / plain cross-entropy) ---
srun python src/cnn_baseline.py \
  --epochs 25 \
  --batch-size 256 \
  --validation-split 0.20 \
  --seed 0

echo "======================================================"
echo "Job finished at $(date)"
echo "======================================================"


