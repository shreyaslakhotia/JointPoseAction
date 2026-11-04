#!/usr/bin/env bash
#SBATCH --job-name=seg-stability
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/projects/jointposeaction/logs/seg-stability-%j.out
#SBATCH --error=/projects/jointposeaction/logs/seg-stability-%j.err
set -euo pipefail
umask 077
cd /projects/jointposeaction/repo
source scripts/cluster/env.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /projects/jointposeaction/experiments/segmentation_pilot/staging
"$JPA_ROOT/environments/sam3-pilot/bin/python" -m unittest discover -s tests \
  -p 'test_segmentation_pilot_stability.py' -v
"$JPA_ROOT/environments/sam3-pilot/bin/python" -m segmentation_pilot.stability \
  --survey "$JPA_ROOT/experiments/segmentation_pilot/survey-268336/survey.json" \
  --landmarks "$JPA_ROOT/experiments/segmentation_pilot/staging/landmarks_initial.json" \
  --output "$JPA_ROOT/experiments/segmentation_pilot/stability-$SLURM_JOB_ID"
