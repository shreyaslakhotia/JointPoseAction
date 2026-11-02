#!/usr/bin/env bash
# One GPU allocation: training-only sanity test, then one fresh baseline.
#SBATCH --partition=cluster02
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --gres=gpu:l40:1
#SBATCH --job-name=e0-run
#SBATCH --time=06:00:00
#SBATCH --output=/projects/jointposeaction/logs/%x-%j.out
#SBATCH --error=/projects/jointposeaction/logs/%x-%j.err
set -euo pipefail
umask 077
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /projects/jointposeaction/repo
source scripts/cluster/env.sh
source scripts/cluster/use_environment.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
"$JPA_PYTHON" -m e0.pipeline stage train
