#!/usr/bin/env bash
#SBATCH --job-name=seg-geometry
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/projects/jointposeaction/logs/seg-geometry-%j.out
#SBATCH --error=/projects/jointposeaction/logs/seg-geometry-%j.err
set -euo pipefail
umask 077
cd /projects/jointposeaction/repo
source scripts/cluster/env.sh
source scripts/cluster/use_environment.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /projects/jointposeaction/experiments/segmentation_pilot/staging
"$JPA_PYTHON" -m unittest discover -s tests -p 'test_segmentation_pilot_*.py' -v
"$JPA_PYTHON" -m segmentation_pilot.epipolar \
  --survey "$JPA_ROOT/experiments/segmentation_pilot/survey-268336/survey.json" \
  --output "$JPA_ROOT/experiments/segmentation_pilot/geometry-$SLURM_JOB_ID.json"
