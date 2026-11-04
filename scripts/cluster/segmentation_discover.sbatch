#!/usr/bin/env bash
#SBATCH --job-name=seg-discover
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=/projects/jointposeaction/logs/seg-discover-%j.out
#SBATCH --error=/projects/jointposeaction/logs/seg-discover-%j.err
set -euo pipefail
umask 077
cd /projects/jointposeaction/repo
source scripts/cluster/env.sh
source scripts/cluster/use_environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$JPA_PYTHON" /projects/jointposeaction/experiments/segmentation_pilot/staging/inspect_segmentation_sources.py
