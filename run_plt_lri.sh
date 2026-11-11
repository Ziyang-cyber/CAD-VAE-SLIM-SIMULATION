#!/bin/bash
#SBATCH --job-name=plot_sim_lri
#SBATCH --output=log/plot_sim_lri_%j.out
#SBATCH --error=log/plot_sim_lri_%j.err
#SBATCH --partition=himem
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

set -euo pipefail

CODE_DIR="/home/zzhang/scratch/hawaii_300/CAD_VAE_cricket/simulation_shuyu/code"
PYTHON_SCRIPT="${CODE_DIR}/plot_sim_lri.py"
ROOT="/home/zzhang/scratch/simulation_shuyu/results/lri_ablation/rep_001"

source /mnt/apps/users/zzhang/conda/etc/profile.d/conda.sh
conda activate vae

cd "${CODE_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

echo "Started: $(date)"

python -u "${PYTHON_SCRIPT}" \
    --root "${ROOT}" \
    --beta 1 \
    --cls 40 \
    --tc 40 \
    --cmi 40 \
    --lri_values 0 1 10 20 40 60 80 100 \
    --epochs 200 \
    --warmup 50 \
    --seed 1024

echo "Finished: $(date)"