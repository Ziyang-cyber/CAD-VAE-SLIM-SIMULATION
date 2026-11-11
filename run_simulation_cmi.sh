#!/bin/bash
#SBATCH --job-name=cadvae_sim_cmi_cls
#SBATCH --output=log/cadvae_sim_cmi_cls_%A_%a.out
#SBATCH --error=log/cadvae_sim_cmi_cls_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --array=0-1
#SBATCH --time=64:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

set -euo pipefail

CLS_VALUES=(40)
CMI_VALUES=(500 1000)

# Each CLS setting is paired with all CMI settings.
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
N_CMI=${#CMI_VALUES[@]}
CLS_INDEX=$((TASK_ID / N_CMI))
CMI_INDEX=$((TASK_ID % N_CMI))

LAMBDA_CLS=${CLS_VALUES[$CLS_INDEX]}
CMI=${CMI_VALUES[$CMI_INDEX]}

BETA=1
LAMBDA_TC=40
LAMBDA_LRI=40
MAX_EPOCHS=200
BETA_WARMUP_EPOCHS=50
BATCH_SIZE=32
SEED=1024

REP_NAME="rep_001"
CODE_DIR="/home/zzhang/scratch/hawaii_300/CAD_VAE_cricket/simulation_shuyu/code/CAD_VAE/model"
REP_DIR="/home/zzhang/scratch/simulation_shuyu/data/${REP_NAME}"

RUN_DIR="/home/zzhang/scratch/simulation_shuyu/results/cmi_cls_ablation/${REP_NAME}"
RUN_DIR="${RUN_DIR}/epochs_${MAX_EPOCHS}_warmup_${BETA_WARMUP_EPOCHS}"
SAVE_DIR="${RUN_DIR}/beta_${BETA}_tc_${LAMBDA_TC}_lri_${LAMBDA_LRI}/cls_${LAMBDA_CLS}/cmi_${CMI}/seed_${SEED}"

source /mnt/apps/users/zzhang/conda/etc/profile.d/conda.sh
conda activate vae

cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

mkdir -p "${SAVE_DIR}"

echo "Started: $(date); repetition=${REP_NAME}"
echo "CLS=${LAMBDA_CLS}; CMI=${CMI}; TC=${LAMBDA_TC}; LRI=${LAMBDA_LRI}; beta=${BETA}"
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