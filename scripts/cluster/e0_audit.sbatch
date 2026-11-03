#!/usr/bin/env bash
# Read-only cached-data/checkpoint audit. No downloads, GPU or optimizer.
#SBATCH --partition=cluster02
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --job-name=e0-audit
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/projects/jointposeaction/logs/%x-%j.out
#SBATCH --error=/projects/jointposeaction/logs/%x-%j.err
set -euo pipefail
umask 077
cd /projects/jointposeaction/repo
source scripts/cluster/env.sh
source scripts/cluster/use_environment.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
"$JPA_PYTHON" -m e0.audit \
  --run "$JPA_ROOT/experiments/e0/controlled-baseline" \
  --output "$JPA_ROOT/experiments/e0/audit-$SLURM_JOB_ID"
