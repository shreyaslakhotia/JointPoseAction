#!/usr/bin/env bash
# Existing-file analysis only: no downloads, training or GPU request.
#SBATCH --partition=cluster02
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --job-name=e0-feasibility
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=/projects/jointposeaction/logs/%x-%j.out
#SBATCH --error=/projects/jointposeaction/logs/%x-%j.err
set -euo pipefail
umask 077
cd /projects/jointposeaction/repo
source scripts/cluster/env.sh
source scripts/cluster/use_environment.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
"$JPA_PYTHON" -m unittest discover -s tests -v
"$JPA_PYTHON" -m scripts.data.e0_feasibility \
    --review "$JPA_ROOT/data/annotations/e0-review-242218/inspection.json" \
    --output "$JPA_ROOT/data/annotations/e0-feasibility-$SLURM_JOB_ID.json"
