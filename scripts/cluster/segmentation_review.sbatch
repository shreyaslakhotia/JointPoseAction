#!/usr/bin/env bash
#SBATCH --job-name=seg-review
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/projects/jointposeaction/logs/seg-review-%j.out
#SBATCH --error=/projects/jointposeaction/logs/seg-review-%j.err
set -euo pipefail
umask 077
cd /projects/jointposeaction/repo
source scripts/cluster/env.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$JPA_ROOT/experiments/segmentation_pilot/staging"
: "${JPA_SEGMENTATION_JOB:?Set the completed inference job ID}"
[[ "$JPA_SEGMENTATION_JOB" =~ ^[0-9]+$ ]]
"$JPA_ROOT/environments/sam3-pilot/bin/python" -m segmentation_pilot.review \
  --run "$JPA_ROOT/experiments/segmentation_pilot/sam3-$JPA_SEGMENTATION_JOB" \
  --output "$JPA_ROOT/experiments/segmentation_pilot/review-$SLURM_JOB_ID"
