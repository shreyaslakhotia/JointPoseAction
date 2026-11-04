#!/usr/bin/env bash
#SBATCH --job-name=seg-survey
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=/projects/jointposeaction/logs/seg-survey-%j.out
#SBATCH --error=/projects/jointposeaction/logs/seg-survey-%j.err
set -euo pipefail
umask 077
cd /projects/jointposeaction/repo
source scripts/cluster/env.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /projects/jointposeaction/experiments/segmentation_pilot/staging
"$JPA_ROOT/environments/sam3-pilot/bin/python" -m segmentation_pilot.acquire_frames \
  --discovery "$JPA_ROOT/experiments/segmentation_pilot/discovery-267862/discovery.json" \
  --output "$JPA_ROOT/experiments/segmentation_pilot/survey-$SLURM_JOB_ID" \
  --max-recordings 3 --seconds 3 \
  --max-network-bytes 805306368 --max-output-bytes 536870912
