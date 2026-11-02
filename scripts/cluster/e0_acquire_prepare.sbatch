#!/usr/bin/env bash
# Planned 7.994 GB selective transfer; CPU preparation only, no GPU/training.
#SBATCH --partition=cluster02
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --job-name=e0-prepare
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --output=/projects/jointposeaction/logs/%x-%j.out
#SBATCH --error=/projects/jointposeaction/logs/%x-%j.err
set -euo pipefail
umask 077
cd /projects/jointposeaction/repo
source scripts/cluster/env.sh
source scripts/cluster/use_environment.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
"$JPA_PYTHON" -m scripts.data.e0_acquire_prepare
