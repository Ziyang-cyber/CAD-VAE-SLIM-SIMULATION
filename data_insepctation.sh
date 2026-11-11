#!/bin/bash
#SBATCH --job-name=data_inspecation
#SBATCH --output=data_inspect_%j.out
#SBATCH --error=data_inspect_%j.err
#SBATCH --partition=himem
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail

source /mnt/apps/users/zzhang/conda/etc/profile.d/conda.sh
conda activate vae

python data_inspectation.py