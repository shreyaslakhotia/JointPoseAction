#!/usr/bin/env bash
# Prepared only. Do not submit until dataset validation and training authorization.
#SBATCH --partition=cluster02
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --gres=gpu:l40:1
#SBATCH --job-name=e0-tcn
#SBATCH --time=02:00:00
#SBATCH --output=/projects/jointposeaction/logs/%x-%j.out
#SBATCH --error=/projects/jointposeaction/logs/%x-%j.err
set -euo pipefail
umask 077
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /projects/jointposeaction/repo
source scripts/cluster/env.sh
module load Miniforge3/24.11.3-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$JPA_ENV"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
: "${JPA_E0_DATASET:?Set path to validated prepared dataset.json}"
python -m e0.train --dataset "$JPA_E0_DATASET" --config configs/e0_tcn.json \
    --output "$JPA_ROOT/experiments/e0/train-$SLURM_JOB_ID"
