#!/usr/bin/env bash
#SBATCH --partition=cluster02
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --job-name=e0-evaluate
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/projects/jointposeaction/logs/%x-%j.out
#SBATCH --error=/projects/jointposeaction/logs/%x-%j.err
set -euo pipefail
umask 077
cd /projects/jointposeaction/repo
source scripts/cluster/env.sh
source scripts/cluster/use_environment.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
: "${JPA_E0_DATASET:?Set prepared dataset path}"
: "${JPA_E0_CHECKPOINT:?Set validation-selected checkpoint path}"
"$JPA_PYTHON" -m e0.evaluate --dataset "$JPA_E0_DATASET" --checkpoint "$JPA_E0_CHECKPOINT" \
    --output "$JPA_ROOT/experiments/e0/test-$SLURM_JOB_ID"
