#!/bin/bash
#SBATCH --job-name=cadvae_sim_lri
#SBATCH --output=log/cadvae_sim_cmi_%A_%a.out
#SBATCH --error=log/cadvae_sim_cmi_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --array=0-7
#SBATCH --time=64:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

set -euo pipefail

LRI_VALUES=(0 1 10 20 40 60 80 100)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
LAMBDA_LRI=${LRI_VALUES[$TASK_ID]}

BETA=1
LAMBDA_CLS=40
LAMBDA_TC=40
CMI=40
MAX_EPOCHS=200
BETA_WARMUP_EPOCHS=50
BATCH_SIZE=32
SEED=1024

# Use the directory containing the modified simulation train.py.
CODE_DIR="/home/zzhang/scratch/hawaii_300/CAD_VAE_cricket/simulation_shuyu/code/CAD_VAE/model"
REP_DIR="/home/zzhang/scratch/simulation_shuyu/data/rep_001"

RUN_DIR="/home/zzhang/scratch/simulation_shuyu/results/lri_ablation/rep_001"
RUN_DIR="${RUN_DIR}/epochs_${MAX_EPOCHS}_warmup_${BETA_WARMUP_EPOCHS}"
SAVE_DIR="${RUN_DIR}/beta_${BETA}_cls_${LAMBDA_CLS}_tc_${LAMBDA_TC}_cmi_${CMI}/lri_${LAMBDA_LRI}/seed_${SEED}"
source /mnt/apps/users/zzhang/conda/etc/profile.d/conda.sh
conda activate vae

cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

mkdir -p "${SAVE_DIR}"

echo "Started: $(date); repetition=rep_001; LRI=${LAMBDA_LRI}; CMI=${CMI}"
echo "Output: ${SAVE_DIR}"

python -u "${CODE_DIR}/train.py" \
    --rep_dir "${REP_DIR}" \
    --save_dir "${SAVE_DIR}" \
    --beta "${BETA}" \
    --beta_warmup_epochs "${BETA_WARMUP_EPOCHS}" \
    --lambda_cls "${LAMBDA_CLS}" \
    --lambda_tc "${LAMBDA_TC}" \
    --lambda_cmi "${CMI}" \
    --lambda_lri "${LAMBDA_LRI}" \
    --max_epochs "${MAX_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers 0 \
    --seed "${SEED}" \
    --device cuda

echo "Finished: $(date)"